from Modules.EDA import *
from Modules.DataMod import *
from Modules.Regressions import *

regressions = ordersWithReturns.drop(columns=['Row.ID', 'Order.Date', 'Ship.Date', 'Customer.ID', 'Customer.Name',
                                                  'Product.ID', 'Postal.Code','Region', 'Market', 'Product.Name',
                                                  'Season', 'Year', 'Returned', 'Return Region'])

dummyColumns =  ['Order.ID', 'Ship.Mode', 'Segment', 'City', 'State', 'Country', 'Product.ID', 'Category', 'Sub.Category', 'Order.Priority']

dummies = pd.get_dummies(regressions, columns=dummyColumns)
regressions.drop(columns=dummyColumns, inplace=True)
regressions = ut.appendColumns(dummies, regressions)

performRegressions(regressions)
