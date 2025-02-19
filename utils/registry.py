import os

class Registry:
    """The class for registry of modules."""
    mapping = {
        "models": {},
        "datasets": {},
        "algorithms": {},
        "tasks": {},
    }

    @classmethod
    def register_attack(cls, name=None, force=False):
        r"""Register an attack method to registry with key 'name'

        Args:
            name (str): Key with which the attacker will be registered.
            force (bool): Whether to register when the name has already existed in registry.
        """

        def wrap(attack):
            registerd_name = attack.__name__ if name is None else name
            if registerd_name in cls.mapping["algorithms"] and not force:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        registerd_name, cls.mapping["algorithms"][registerd_name]
                    )
                )

            cls.mapping["algorithms"][registerd_name] = attack
            return attack

        return wrap

    @classmethod
    def register_model(cls, name=None, force=False):
        r"""Register a model to registry with key 'name'

        Args:
            name (str): Key with which the attacker will be registered.
            force (bool): Whether to register when the name has already existed in registry.
        """

        def wrap(model):
            registerd_name = model.__name__ if name is None else name
            if registerd_name in cls.mapping["models"] and not force:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        registerd_name, cls.mapping["models"][registerd_name]
                    )
                )
            cls.mapping["models"][registerd_name] = model
            return model

        return wrap
    
    @classmethod
    def register_data(cls, name=None, force=False):
        r"""Register a dataloader to registry with key 'name'

        Args:
            name (str): Key with which the attacker will be registered.
            force (bool): Whether to register when the name has already existed in registry.
        """

        def wrap(datasets):
            registerd_name = datasets.__name__ if name is None else name
            if registerd_name in cls.mapping["datasets"] and not force:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        registerd_name, cls.mapping["datasets"][registerd_name]
                    )
                )
            cls.mapping["datasets"][registerd_name] = datasets
            return datasets

        return wrap
    
    @classmethod
    def register_task(cls, name=None, force=False):
        r"""Register a task to registry with key 'name'

        Args:
            name (str): Key with which the attacker will be registered.
            force (bool): Whether to register when the name has already existed in registry.
        """

        def wrap(tasks):
            registerd_name = tasks.__name__ if name is None else name
            if registerd_name in cls.mapping["tasks"] and not force:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        registerd_name, cls.mapping["tasks"][registerd_name]
                    )
                )
            cls.mapping["tasks"][registerd_name] = tasks
            return tasks

        return wrap
    
    @classmethod
    def get_task(cls, name):
        '''Get a task by given name.'''
        if cls.mapping["tasks"].get(name, None):
            return cls.mapping["tasks"].get(name)
        raise KeyError(f'{name} is not registered!')
    
    @classmethod
    def get_data(cls, name):
        '''Get a datasetloader by given name.'''
        if cls.mapping["datasets"].get(name, None):
            return cls.mapping["datasets"].get(name)
        raise KeyError(f'{name} is not registered!')

    @classmethod
    def get_model(cls, name):
        '''Get a model object by given name.'''
        if cls.mapping["models"].get(name, None):
            return cls.mapping["models"].get(name)
        raise KeyError(f'{name} is not registered!')
    
    @classmethod
    def get_attack(cls, name):
        '''Get a attack method by given name.'''
        if cls.mapping["algorithms"].get(name, None):
            return cls.mapping["algorithms"].get(name)
        raise KeyError(f'{name} is not registered!')

registry = Registry()