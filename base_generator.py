import abc
# implement an interface generator class 

class BaseGenerator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def generate_candidates(self, category_ids: list[int], candidate_num: int, *args, **kwargs):
        """ given a category_id and other parameters, 
            return a list of candidate category_ids in sorted order
        """
        pass 
    
    @abc.abstractmethod
    def generate_baseline_candidates(self, candidate_num, *args, **kwargs):
        """ baseline candidate generation method 
        """
        pass