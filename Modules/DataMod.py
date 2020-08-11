## Part II: Machine Learning and Business Use Case
from Modules.EDA import *

# Now your manager has a basic understanding of why customers returned orders. Next, he wants you to use machine learning
# to predict which orders are most likely to be returned. In this part, you will generate several features based on our previous findings and your manager's requirements.

### Problem 4: Feature Engineering
#### Step 1: Create the dependent variable
# - First of all, we need to generate a categorical variable which indicates whether an order has been returned or not.
# - ***Hint:*** the returned orders’ IDs are contained in the dataset “returns”
# Already done in EDA

#### Step 2:
# - Your manager believes that **how long it took the order to ship** would affect whether the customer would return it or not.
# - He wants you to generate a feature which can measure how long it takes the company to process each order.
# - ***Hint:*** Process.Time = Ship.Date - Order.Date
#ordersWithReturns['Process.Time'] = ordersWithReturns[]

ut.stringToDatetime(ordersWithReturns, ['Ship.Date', 'Order.Date'], inplace=True)

#ut.dateFormat(ordersWithReturns)
ordersWithReturns['Process.Time'] = ordersWithReturns['Ship.Date'] - ordersWithReturns['Order.Date']
ordersWithReturns['Process.Time'] = ordersWithReturns['Process.Time'].apply(lambda x: x.days)

#### Step 3:

# - If a product has been returned before, it may be returned again.
# - Let us generate a feature indictes how many times the product has been returned before.
# - If it never got returned, we just impute using 0.
# - ***Hint:*** Group by different Product.ID
returnsByProduct = ordersWithReturns[ordersWithReturns['Returned'] == 'Yes'].groupby('Product.ID')[['Returned']].count()
returnsByProduct.reset_index(inplace=True)
ordersWithReturns = ordersWithReturns.merge(returnsByProduct, on='Product.ID', how='left')
ordersWithReturns.rename(columns={'Returned_y': 'Returned_Count', 'Returned_x':'Returned'}, inplace=True)
ordersWithReturns['Returned_Count'] = ordersWithReturns['Returned_Count'].fillna(0)
ut.convertColumns(ordersWithReturns, columns=['Returned_Count'], newType=int)
print('Finished Data Mods.')