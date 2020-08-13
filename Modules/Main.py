from Modules.EDA import EDA
from Modules.DataMod import DataMod
from Modules.Regressions import performRegressions
import pandas as pd
import Modules.Util as ut


orders = EDA()
fullDF = DataMod(orders)

regressions = fullDF.drop(columns=['Row.ID', 'Order.Date', 'Ship.Date', 'Customer.ID', 'Customer.Name',
                                              'Postal.Code','Region', 'Market', 'Product.Name', 'Season',
                                              'Year', 'Returned_Count', 'Return Region',
                                   'City', 'State', 'Order.ID', 'Product.ID'])

dummyColumns =  ['Ship.Mode', 'Segment', 'Country', 'Category', 'Sub.Category', 'Order.Priority']

dummies = pd.get_dummies(regressions, columns=dummyColumns, drop_first=True)
performRegressions(dummies)
