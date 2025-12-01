import graph_tool as gt
import os
import pathlib
import warnings

import torch
torch.cuda.empty_cache()
import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.warnings import PossibleUserWarning

from src import utils
from src.metrics.abstract_metrics import TrainAbstractMetricsDiscrete, TrainAbstractMetrics

from src.diffusion_model import LiftedDenoisingDiffusion
from src.diffusion_model_discrete import DiscreteDenoisingDiffusion
from src.diffusion.extra_features import DummyExtraFeatures, ExtraFeatures


warnings.filterwarnings("ignore", category=PossibleUserWarning)


def get_resume(cfg, model_kwargs):
    """
    Resume from checkpoint for testing only.
    Loads previous config without allowing updates.
    """
    saved_cfg = cfg.copy()
    name = cfg.general.name + '_resume'
    resume = cfg.general.test_only
    
    if cfg.model.type == 'discrete':
        model = DiscreteDenoisingDiffusion.load_from_checkpoint(resume, **model_kwargs)
    else:
        model = LiftedDenoisingDiffusion.load_from_checkpoint(resume, **model_kwargs)
    
    cfg = model.cfg
    cfg.general.test_only = resume
    cfg.general.name = name
    cfg = utils.update_config_with_new_keys(cfg, saved_cfg)
    
    return cfg, model


def get_resume_adaptive(cfg, model_kwargs):
    """
    Resume training from checkpoint with support for configuration changes.
    Handles dimension changes in input/output layers for transfer learning.
    """
    saved_cfg = cfg.copy()
    
    # Get checkpoint path
    current_path = os.path.dirname(os.path.realpath(__file__))
    root_dir = current_path.split('outputs')[0]
    resume_path = os.path.join(root_dir, cfg.general.resume)
    
    # Check if dimensions have changed by comparing dataset info
    checkpoint = torch.load(resume_path, map_location='cpu')
    old_hyperparams = checkpoint.get('hyper_parameters', {})
    
    dims_changed = False
    if 'dataset_infos' in old_hyperparams:
        old_input_dims = old_hyperparams['dataset_infos'].input_dims
        new_input_dims = model_kwargs['dataset_infos'].input_dims
        
        if old_input_dims != new_input_dims:
            dims_changed = True
            print("\n" + "="*80)
            print("⚠️  INPUT DIMENSIONS CHANGED - Using custom loading")
            print(f"Old dims: {old_input_dims}")
            print(f"New dims: {new_input_dims}")
            print("="*80 + "\n")
    
    if dims_changed:
        # Use custom loading that handles dimension changes
        return get_resume_adaptive_with_new_dims(cfg, model_kwargs, resume_path, checkpoint)
    else:
        # Standard loading when dimensions match
        if cfg.model.type == 'discrete':
            model = DiscreteDenoisingDiffusion.load_from_checkpoint(resume_path, **model_kwargs)
        else:
            model = LiftedDenoisingDiffusion.load_from_checkpoint(resume_path, **model_kwargs)
        
        new_cfg = model.cfg
        
        # Override with new config values
        for category in cfg:
            for arg in cfg[category]:
                new_cfg[category][arg] = cfg[category][arg]
        
        new_cfg.general.resume = resume_path
        new_cfg.general.name = new_cfg.general.name + '_resume'
        new_cfg = utils.update_config_with_new_keys(new_cfg, saved_cfg)
        
        return new_cfg, model


def get_resume_adaptive_with_new_dims(cfg, model_kwargs, resume_path, checkpoint):
    """
    Custom loading for transfer learning with changed dimensions.
    Loads transformer weights but reinitializes input/output MLPs.
    """
    # Create new model with new dimensions
    if cfg.model.type == 'discrete':
        model = DiscreteDenoisingDiffusion(cfg=cfg, **model_kwargs)
    else:
        model = LiftedDenoisingDiffusion(cfg=cfg, **model_kwargs)
    
    # Get state dictionaries
    model_state = model.state_dict()
    checkpoint_state = checkpoint['state_dict']
    
    # Filter keys to load only transformer layers
    keys_to_load = {}
    skipped_keys = []
    
    for key, value in checkpoint_state.items():
        # Load only transformer layers, skip input/output MLPs
        if 'model.tf_layers' in key:
            # Remove 'model.' prefix if needed for compatibility
            new_key = key.replace('model.', '') if key.startswith('model.') else key
            
            if new_key in model_state and model_state[new_key].shape == value.shape:
                keys_to_load[new_key] = value
                print(f"✓ Loading: {key}")
            else:
                skipped_keys.append(key)
                print(f"✗ Skipping: {key} (shape mismatch or not found)")
        
        elif 'model.mlp_in' in key or 'model.mlp_out' in key:
            skipped_keys.append(key)
            print(f"✗ Skipping (will reinitialize): {key}")
    
    # Load the filtered state dict (only transformer layers)
    model.model.load_state_dict(keys_to_load, strict=False)
    
    print(f"\n{'='*80}")
    print(f"✓ Loaded {len(keys_to_load)} transformer layer weights from checkpoint")
    print(f"✓ Reinitialized {len(skipped_keys)} input/output MLP weights")
    print(f"{'='*80}\n")
    
    # Update config
    new_cfg = cfg.copy()
    new_cfg.general.resume = resume_path
    new_cfg.general.name = new_cfg.general.name + '_resume'
    
    # Apply freezing if specified
    if hasattr(cfg.general, 'freeze_transformer') and cfg.general.freeze_transformer:
        model.freeze_transformer_layers()
    
    return new_cfg, model


@hydra.main(version_base='1.3', config_path='../configs', config_name='config')
def main(cfg: DictConfig):
    dataset_config = cfg["dataset"]

    # Setup dataset-specific components
    if dataset_config["name"] in ['sbm', 'comm20', 'planar']:
        from src.datasets.spectre_dataset import SpectreGraphDataModule, SpectreDatasetInfos
        from src.analysis.spectre_utils import PlanarSamplingMetrics, SBMSamplingMetrics, Comm20SamplingMetrics
        from src.analysis.visualization import NonMolecularVisualization

        datamodule = SpectreGraphDataModule(cfg)
        
        # Select appropriate sampling metrics
        if dataset_config['name'] == 'sbm':
            sampling_metrics = SBMSamplingMetrics(datamodule)
        elif dataset_config['name'] == 'comm20':
            sampling_metrics = Comm20SamplingMetrics(datamodule)
        else:
            sampling_metrics = PlanarSamplingMetrics(datamodule)

        dataset_infos = SpectreDatasetInfos(datamodule, dataset_config)
        train_metrics = TrainAbstractMetricsDiscrete() if cfg.model.type == 'discrete' else TrainAbstractMetrics()
        visualization_tools = NonMolecularVisualization()

        # Setup extra features (e.g., eigenvalues, Ricci curvature)
        if cfg.model.type == 'discrete' and cfg.model.extra_features is not None:
            extra_features = ExtraFeatures(
                cfg.model.extra_features, 
                dataset_info=dataset_infos,
                ricci_alpha=getattr(cfg.model, 'ricci_alpha', 0.5)
            )
        else:
            extra_features = DummyExtraFeatures()
        
        domain_features = DummyExtraFeatures()

        dataset_infos.compute_input_output_dims(
            datamodule=datamodule, 
            extra_features=extra_features,
            domain_features=domain_features
        )

        model_kwargs = {
            'dataset_infos': dataset_infos, 
            'train_metrics': train_metrics,
            'sampling_metrics': sampling_metrics, 
            'visualization_tools': visualization_tools,
            'extra_features': extra_features, 
            'domain_features': domain_features
        }

    elif dataset_config["name"] in ['qm9', 'guacamol', 'moses']:
        from src.metrics.molecular_metrics import TrainMolecularMetrics, SamplingMolecularMetrics
        from src.metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete
        from src.diffusion.extra_features_molecular import ExtraMolecularFeatures
        from src.analysis.visualization import MolecularVisualization

        if dataset_config["name"] == 'qm9':
            from src.datasets import qm9_dataset
            datamodule = qm9_dataset.QM9DataModule(cfg)
            dataset_infos = qm9_dataset.QM9infos(datamodule=datamodule, cfg=cfg)
            train_smiles = qm9_dataset.get_train_smiles(
                cfg=cfg, 
                train_dataloader=datamodule.train_dataloader(),
                dataset_infos=dataset_infos, 
                evaluate_dataset=False
            )
        
        elif dataset_config['name'] == 'guacamol':
            from src.datasets import guacamol_dataset
            datamodule = guacamol_dataset.GuacamolDataModule(cfg)
            dataset_infos = guacamol_dataset.Guacamolinfos(datamodule, cfg)
            train_smiles = None

        elif dataset_config.name == 'moses':
            from src.datasets import moses_dataset
            datamodule = moses_dataset.MosesDataModule(cfg)
            dataset_infos = moses_dataset.MOSESinfos(datamodule, cfg)
            train_smiles = None
        else:
            raise ValueError("Dataset not implemented")

        # Setup extra features for molecular data
        if cfg.model.type == 'discrete' and cfg.model.extra_features is not None:
            extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
            domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
        else:
            extra_features = DummyExtraFeatures()
            domain_features = DummyExtraFeatures()

        dataset_infos.compute_input_output_dims(
            datamodule=datamodule, 
            extra_features=extra_features,
            domain_features=domain_features
        )

        if cfg.model.type == 'discrete':
            train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)
        else:
            train_metrics = TrainMolecularMetrics(dataset_infos)

        sampling_metrics = SamplingMolecularMetrics(dataset_infos, train_smiles)
        visualization_tools = MolecularVisualization(cfg.dataset.remove_h, dataset_infos=dataset_infos)

        model_kwargs = {
            'dataset_infos': dataset_infos, 
            'train_metrics': train_metrics,
            'sampling_metrics': sampling_metrics, 
            'visualization_tools': visualization_tools,
            'extra_features': extra_features, 
            'domain_features': domain_features
        }
    else:
        raise NotImplementedError("Unknown dataset {}".format(cfg["dataset"]))

    # Handle different modes: test only, resume, or fresh training
    if cfg.general.test_only:
        # Load model for testing only
        cfg, _ = get_resume(cfg, model_kwargs)
        os.chdir(cfg.general.test_only.split('checkpoints')[0])
    elif cfg.general.resume is not None:
        # Resume training with possible config overrides
        cfg, _ = get_resume_adaptive(cfg, model_kwargs)
        os.chdir(cfg.general.resume.split('checkpoints')[0])

    utils.create_folders(cfg)

    # Create model
    if cfg.model.type == 'discrete':
        model = DiscreteDenoisingDiffusion(cfg=cfg, **model_kwargs)
    else:
        model = LiftedDenoisingDiffusion(cfg=cfg, **model_kwargs)

    # Setup callbacks
    callbacks = []
    if cfg.train.save_model:
        # Save top 5 checkpoints based on validation NLL
        checkpoint_callback = ModelCheckpoint(
            dirpath=f"checkpoints/{cfg.general.name}",
            filename='{epoch}',
            monitor='val/epoch_NLL',
            save_top_k=5,
            mode='min',
            every_n_epochs=1
        )
        # Always save last checkpoint
        last_ckpt_save = ModelCheckpoint(
            dirpath=f"checkpoints/{cfg.general.name}", 
            filename='last', 
            every_n_epochs=1
        )
        callbacks.append(last_ckpt_save)
        callbacks.append(checkpoint_callback)

    # Add EMA callback if specified
    if cfg.train.ema_decay > 0:
        ema_callback = utils.EMA(decay=cfg.train.ema_decay)
        callbacks.append(ema_callback)

    name = cfg.general.name
    if name == 'debug':
        print("[WARNING]: Run is called 'debug' -- it will run with fast_dev_run. ")

    # Setup trainer
    use_gpu = cfg.general.gpus > 0 and torch.cuda.is_available()
    trainer = Trainer(
        gradient_clip_val=cfg.train.clip_grad,
        strategy="ddp_find_unused_parameters_true",  # Needed to load old checkpoints
        accelerator='gpu' if use_gpu else 'cpu',
        devices=cfg.general.gpus if use_gpu else 1,
        max_epochs=cfg.train.n_epochs,
        check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
        fast_dev_run=cfg.general.name == 'debug',
        enable_progress_bar=False,
        callbacks=callbacks,
        log_every_n_steps=50 if name != 'debug' else 1,
        logger=[]
    )

    if not cfg.general.test_only:
        # Train and then test
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.general.resume)
        if cfg.general.name not in ['debug', 'test']:
            trainer.test(model, datamodule=datamodule)
    else:
        # Test only mode
        trainer.test(model, datamodule=datamodule, ckpt_path=cfg.general.test_only)
        
        # Optionally evaluate all checkpoints in directory
        if cfg.general.evaluate_all_checkpoints:
            directory = pathlib.Path(cfg.general.test_only).parents[0]
            print("Directory:", directory)
            files_list = os.listdir(directory)
            
            for file in files_list:
                if '.ckpt' in file:
                    ckpt_path = os.path.join(directory, file)
                    if ckpt_path == cfg.general.test_only:
                        continue
                    print("Loading checkpoint", ckpt_path)
                    trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == '__main__':
    main()