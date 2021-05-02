try:
    import Box2D
    from .lunar_lander import LunarLander
    from .lunar_lander import LunarLanderContinuous
    from .bipedal_walker import BipedalWalker, BipedalWalkerHardcore
    from .car_racing import CarRacing
except ImportError:
    Box2D = None
