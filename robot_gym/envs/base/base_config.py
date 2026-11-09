import copy
import inspect


class BaseConfig:
    def __init__(self) -> None:
        """Materialize sections and their values so separate runs own their settings."""
        self.init_member_classes(self)

    @staticmethod
    def init_member_classes(obj):
        for key in dir(obj):
            if key.startswith("_"):
                continue

            var = getattr(obj, key)

            if inspect.isclass(var):
                i_var = var()
                setattr(obj, key, i_var)
                BaseConfig.init_member_classes(i_var)
            elif not callable(var):
                setattr(obj, key, copy.deepcopy(var))

    def to_dict(self):
        return self._recursive_to_dict(self)

    @staticmethod
    def _recursive_to_dict(obj):
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj

        if isinstance(obj, (list, tuple)):
            return [BaseConfig._recursive_to_dict(x) for x in obj]

        if isinstance(obj, dict):
            return {
                key: BaseConfig._recursive_to_dict(value)
                for key, value in obj.items()
            }

        if not hasattr(obj, "__dict__"):
            return obj

        result = {}
        for key in dir(obj):
            if key.startswith("_"):
                continue

            value = getattr(obj, key)

            if inspect.ismethod(value) or inspect.isfunction(value):
                continue

            result[key] = BaseConfig._recursive_to_dict(value)

        return result
