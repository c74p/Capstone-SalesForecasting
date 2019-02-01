from tdda.referencetest.pytestconfig import (pytest_addoption,
                                             pytest_collection_modifyitems,
                                             set_default_data_location,
                                             ref)

set_default_data_location('data/processed')
