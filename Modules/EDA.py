import pandas as pd
import matplotlib.pyplot as plt
import Modules.Util as ut
import datetime
from dateutil.relativedelta import relativedelta
from datetime import date

### Problem 1: Dataset Import & Cleaning


orders = pd.read_csv('Data/Orders.csv')
returns = pd.read_csv('Data/Returns.csv')

# Check **"Profit"** and **"Sales"** in the dataset, convert these two columns to numeric type.
ut.toNumericString(orders, ['Sales', 'Profit'])
ut.convertColumns(orders,['Sales', 'Profit'],float)

# ### Problem 2: Inventory Management
# - Retailers that depend on seasonal shoppers have a particularly challenging job when it comes to inventory management.
# Your manager is making plans for next year's inventory.
# - He wants you to answer the following questions:
#     1. Is there any seasonal trend of inventory in the company?
orders['Season'] = orders['Order.Date'].apply(lambda x: x.split('/')[0])
orders['Season'] = orders['Season'].apply(ut.dateToSeason)

orders['Year'] = orders['Order.Date'].apply(lambda x: '20'+x.split('/')[2])
seasonCounts = orders.groupby(['Season']).sum()[['Quantity']]
seasonCounts.plot(kind="bar", figsize=(20,10))
plt.savefig('Plots/Orders by Season')

seasonAndProductCounts = orders.groupby(['Season','Order.ID']).count()


seasonAndYear = orders.groupby(['Season', 'Year']).sum()[['Quantity']]
seasonAndYear.unstack().plot(kind="bar", figsize=(20,10))
plt.savefig('Plots/Orders by Season and Year')

yearAndSeason = orders.groupby(['Year', 'Season']).sum()[['Quantity']]
yearAndSeason.unstack().plot(kind="bar", figsize=(20,10))
plt.savefig('Plots/Orders by Year and Season')

# Yes, there is a seasonal trend. Sales are highest in Fall, and lowest in Spring


#     2. Is the seasonal trend the same for different categories?
categoryAndSeason = orders.groupby(['Category', 'Season']).sum()[['Quantity']]
categoryAndSeason.unstack().plot(kind="bar", figsize=(20,10))
plt.savefig('Plots/Orders by Season and Category')

# Yes, the trend is consistent across categories. The greater the overall sales, the bigger the gap between seasons.

#
# - ***Hint:*** For each order, it has an attribute called `Quantity` that indicates the number of product in the order.
# If an order contains more than one product, there will be multiple observations of the same order.
#
#
# ### Problem 3: Why did customers make returns?
# - Your manager required you to give a brief report (**Plots + Interpretations**) on returned orders.
#
# 	1. How much profit did we lose due to returns each year?
ordersWithReturns = orders.merge(returns, left_on='Order.ID', right_on='Order ID', how='left')
ordersWithReturns = ordersWithReturns.rename(columns={'Region_y': 'Return Region', 'Region_x': 'Region'}).drop(columns='Order ID')

ordersWithReturns['Returned'].fillna('No', inplace=True)

returnedOrders = ordersWithReturns[ordersWithReturns['Returned'] == 'Yes']
#calculate loss per return
returnedOrders['Return_Profit_Loss'] = returnedOrders['Profit']
returnedOrders['Return_Profit_Loss'] = returnedOrders['Return_Profit_Loss'].apply(lambda x: x * -2 if x < 0 else x)

plt.clf()
totalReturnLoss = returnedOrders.groupby(['Year'])['Return_Profit_Loss'].sum()
totalReturnLoss.plot(kind='bar', figsize=(20,10), legend=False)
plt.savefig('Plots/Losses from Returns by Year')


#
# 	2. How many customer returned more than once? more than 5 times?

returnsByCustomer = returnedOrders.groupby('Customer.ID')['Returned'].count().sort_values(ascending=False)
returnsGT1 = returnsByCustomer[returnsByCustomer > 1].count() #547
returnsGT5 = returnsByCustomer[returnsByCustomer > 5].count() #46

#
# 	3. Which regions are more likely to return orders?

regionReturns = returns.groupby('Region').count().sort_values(by='Order ID', ascending=True)
regionReturns.plot(kind="barh", figsize=(20,10))
plt.savefig('Plots/Returns by Region')

# Western Europe and Central America have the most returns.



# 	4. Which categories (sub-categories) of products are more likely to be returned?
plt.clf()
returnsByCategory = returnedOrders.groupby('Sub.Category')['Returned'].count().sort_values(ascending=True)
returnsByCategory.plot(kind="barh", figsize=(20,10))
plt.savefig('Plots/Returns by Category')

#Binders, art, and storage products are the most likely to be returned.
print('Finished EDA')