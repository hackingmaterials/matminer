import itertools

from pymatgen.core.composition import Composition
from pymatgen.core.periodic_table import Element

from matminer.design.base import BaseGenerator


class CompositionGenerator(BaseGenerator):
    """Generate a range of compositions"""

    def __init__(self, elements, min_elements=1, max_elements=3, spacing=50):
        """Initialize the generator

        Args:
            elements ([str or Element]): List of elements to use in generator
            min_elements (int): Minimum number of elements to use in composition
            max_elements (int): Maximum number of elements to use in composition
            spacing (int): Number of entries to generate for a binary system.
                Controls spacing between compositions.
            """

        self.elements = [Element(e) for e in elements]
        self.min_elements = min_elements
        self.max_elements = max_elements
        self.spacing = spacing

    def generate_entries(self):
        # Convert elements to a set
        elements = set([Element(x) for x in self.elements])

        # Generate the compositions
        for c in range(self.min_elements, self.max_elements + 1):
            stoichs = self._generate_stoichiometries(c, self.spacing)
            for e, s in itertools.product(itertools.combinations(elements, c),
                                          stoichs):
                yield Composition(zip(e, s))

    def _generate_stoichiometries(self, count, sum):
        """Generate lists of N counting numbers that add up to a certain sum

        Args:
            count (int): Number of integers
            sum (int): Target sum
        Returns:
            Generator of [int] where all entries are >0 and the sum is target
        """

        if count == 1:
            yield (sum,)
        else:
            for start in range(1, sum):
                for end in self._generate_stoichiometries(count-1, sum-start):
                    yield (start,) + end
