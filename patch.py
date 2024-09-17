def warnings():
    import warnings
    warnings.simplefilter("ignore")

    original_filterwarnings = warnings.filterwarnings
    def _filterwarnings(*args, **kwargs):
        return original_filterwarnings(*args, **{**kwargs, 'append':True})
    warnings.filterwarnings = _filterwarnings

def ddp():
    #patch use_ddp
    from pytorch_lightning.trainer.connectors.accelerator_connector import _LITERAL_WARN, AcceleratorConnector
    AcceleratorConnector.use_ddp = lambda self: self._strategy_type in (
            _StrategyType.BAGUA,
            _StrategyType.DDP,
            _StrategyType.DDP_SPAWN,
            _StrategyType.DDP_SHARDED,
            _StrategyType.DDP_SHARDED_SPAWN,
            _StrategyType.DDP_FULLY_SHARDED,
            _StrategyType.DEEPSPEED,
            _StrategyType.TPU_SPAWN,
        )

def all():
    warnings()
    # ddp()

