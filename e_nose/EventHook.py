from typing import Callable, List


# Modeled after Michael Foord's Event Pattern:
# http://www.voidspace.org.uk/python/weblog/arch_d7_2007_02_03.shtml#e616
class EventHook(object):
    """
        Event Hook that allows adding and calling methods similar to C# delegates / Java EventListeners
        Usage:
        in __init__:
        self.some_hook = EventHook()

        in some_method:
        # Call all attached handlers
        self.some_hook()

        Outside:
        object.some_hook += my_handler_function
    """

    def __init__(self):
        self.__delegates: List[Callable] = []

    def __iadd__(self, delegate: Callable):
        """ Add a delegate to the hook to be called """
        self.__delegates.append(delegate)
        return self

    def __isub__(self, delegate: Callable):
        """ Remove a delegate from the hook to be called """
        self.__delegates.remove(delegate)
        return self

    def __call__(self, *args, **kwargs):
        """ Call all delegates that have been added to the hook, returns the list of the results of all delegates """
        ret = []
        for delegate in self.__delegates:
            ret.append(delegate(*args, **kwargs))
        return ret

    def clear(self):
        """ Remove all delegates from the hook """
        self.__delegates = []
