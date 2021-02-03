'''Bla bla

'''
from dataclasses import dataclass

@dataclass
class OrganelleMeta:
    name : str
    within_nucleus : bool

organelle_meta = [
    OrganelleMeta('Nucleoplasm', True),
    OrganelleMeta('Nuclear membrane', True),
    OrganelleMeta('Nucleoli', True),
    OrganelleMeta('Nucleoli fibrillar center', True),
    OrganelleMeta('Nuclear speckles', True),
    OrganelleMeta('Nuclear bodies', True),
    OrganelleMeta('Endoplasmic reticulum', False),
    OrganelleMeta('Golgi apparatus', False),
    OrganelleMeta('Intermediate filaments', False),
    OrganelleMeta('Actin filaments', False),
    OrganelleMeta('Microtubules', False),
    OrganelleMeta('Mitotic spindle', False),
    OrganelleMeta('Centrosome', False),
    OrganelleMeta('Plasma membrane', False),
    OrganelleMeta('Mitochondria', False),
    OrganelleMeta('Aggresome', False),
    OrganelleMeta('Cytosol', False),
    OrganelleMeta('Vesicles and punctate cytosolic patterns', False)
]
