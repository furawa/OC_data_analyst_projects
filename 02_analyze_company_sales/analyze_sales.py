#!/usr/bin/env python
# coding: utf-8

# # ANALYZE DATA
# import the cleaned datasets
# Load all the needed packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import scipy.stats as sci
import time
import statsmodels.api as sm
# get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_palette("colorblind")
sns.set_context("talk")
sns.set(style = "white", rc={'figure.figsize':(11.7,8.27)})

## Import The Dataset
print("Import the cleaned dataset for the analysis")
sales_clean = pd.read_csv("clean_datasets/sales_clean.csv", 
                          dtype={"category":"category", "gender":"category"},
                          parse_dates=["date"])
print("sales dataframe: \n", sales_clean)

print("Summary of the Sales:\n")

# Total sales since the beginning
total_sales = sales_clean.price.sum()
# Total sales in 2021
total_sales_2021 = sales_clean[sales_clean.date.dt.year == 2021].price.sum()
# Total sales in 2022
total_sales_2022 = sales_clean[sales_clean.date.dt.year == 2022].price.sum()
# Total products sold
total_products = len(sales_clean.id_prod.unique())
total_transactions = len(sales_clean.session_id.unique())
print("From March 2021 to February 2022 the company make a total sale of ${:,}.\nIn 2021 the total sale was ${:,},\
and just in January and February 2022 the total sale is ${:,}. This means that in just 2 months we make {}% the sales of 2021.\n\
{} products have been sold with a total of {:,} transactions.".format(round(total_sales), round(total_sales_2021),
                                                                    round(total_sales_2022),
                                                                    round(100*total_sales_2022/total_sales_2021),
                                                                    total_products, total_transactions))


print("Univariate Analysis") 
# 
# ### **What is the Average Price of the Products?**
sales_clean[sales_clean.date == sales_clean.date.min()]

# We have an average price of **17.22** but with a standard deviation of **17.85** which means that there are too much variability between the prices, there are probably outliers. Let's check for the IQR.  

# IQR of the price
def iqr(column) -> float:
    """return the iqr of a given column"""
    return column.quantile(.75) - column.quantile(.25)
# IQR of the price
iqr(sales_clean.price)


# We have an IQR of around **10**, the variability is high. Let's make a plot.

print("Boxplot to see the variability")
sns.boxplot("price", data = sales_clean)
plt.title("Variability of the Prices", fontsize = 24)
plt.xlabel("Price ($)", fontsize = 18, fontweight = "bold", fontstyle = "italic")
plt.show()
plt.savefig("images/boxplot_price.png");
print("The figure has been saved in images folder")
time.sleep(2)


# As we can see there are many outliers. Let's try to find them all. 

# Find all the positive outliers, Q3 + 1.5IQR
outliers = sales_clean.price.quantile(0.75)+1.5*iqr(sales_clean.price)
sales_clean.price[sales_clean.price > outliers].sort_values()


print("All the price between **34.77** to **300** are outliers.") 

print("Distribution of the non outliers values")
sales_clean_no_outliers = sales_clean[sales_clean.price < outliers] # Compute the dataframe of the non outliers
bins = int(round(1 + np.log2(sales_clean_no_outliers.shape[0]))) # Optimal number of bins
sns.distplot(sales_clean_no_outliers.price, bins = bins) # Distribution plot of the non outliers values
plt.axvline(sales_clean_no_outliers.price.mean(), color = "red") # Plot vertical line of the mean
plt.axvline(sales_clean_no_outliers.price.median(), color = "orange") # Plot the vertical line of the median
plt.legend(["Mean", "Median"], fontsize = 14)
plt.title("Distribution of the Price without Outliers", fontsize = 24, fontstyle = "italic")
plt.xlabel("Price ($)", fontsize = 18, fontweight = "bold")
plt.show()
plt.savefig("images/hist_price_no_out.png");
print("The figure has been saved in images folder")
time.sleep(2)

sales_clean_no_outliers.price.agg([np.mean, np.median])


print("If we remove all the outliers the mean and median are almost the same and are around $13.") 

print("check the proportion of outliers in the dataframe")
sales_clean_outliers = sales_clean.price[sales_clean.price > outliers]
prop_outliers = len(sales_clean_outliers) / len(sales_clean) * 100
prop_outliers


#The outliers represent about **6%** of the number of sales. Let see what it represent as value


# total sales for the outliers
outliers_total_sales = sales_clean_outliers.sum() / sales_clean.price.sum() * 100
outliers_total_sales


print("Even if the outliers represent just **6%** of the sales, they represent **25%** of the total purchases amount.")
time.sleep(2)
print("mean, median and mode of the price in the dataframe")
print("mean : ",sales_clean.price.mean())
print("median : ",sales_clean.price.median())
print("mode : ",sales_clean.price.mode()[0])


print("We confirm here that most of the sales are between **0 and 16**, with highest sales at **13.9**")
time.sleep(2)
print("Concentration of sales - Lorenz Curve")


# Plot the lorenz curve
n = len(sales_clean.price) #size of the sample
lorenz = np.cumsum(np.sort(sales_clean.price) / sales_clean.price.sum()) #result will be an array
lorenz = np.append([0], lorenz) #values of the y axis on the curve, the curve will begin at 0

# plt.axes().axis("equal") #x and y axis will have the same length
xaxis = np.linspace(-1/n, 1+1/n, n+1) #values of the x axis on the curve
plt.plot(xaxis, lorenz, drawstyle = "steps-post") # Plot lorenz curve
plt.plot(xaxis, xaxis, linestyle="--", color = "green") # Plot the 1st bissector
plt.xlim(0,1) # To make the axis starting at 0
plt.ylim(0,1)
plt.ylabel("Cumulative Ratio of Purchases amount", fontsize = 18, fontweight = "bold")
plt.xlabel("Cumulative Ratio of Sales", fontsize = 18, fontweight = "bold")
plt.axvline(0.843, ymax = 0.6, linestyle = "--", color = "r") # Plot vertical and horizontal line
plt.axhline(0.6, xmax = 0.84, linestyle = "--", color = "r")  #
plt.title("Concentration Analysis - Sales", fontsize = 24, fontweight = "bold", fontstyle = "italic")
plt.text(s = "Line of Equality", y = 0.35, x = 0.32, rotation = 35, fontsize = 16, color = "green")
plt.text(s = "Lorenz Curve", y = 0.32, x = 0.55, rotation = 35, fontsize = 16, color = "b")
plt.show()
plt.savefig("images/lorenz_curve.png");
print("The figure has been saved in images folder")
time.sleep(2)


# #### **Gini Coefficient**


print("compute gini coefficient")
AUC = (lorenz.sum() - lorenz[-1]/2 - lorenz[0]/2)/n # Area under the curve
S = 0.5 - AUC # Area between the curve and the 1st bissector
gini = round(2*S,1)*100


print("""We have a gini coefficient of around 40%, which means there is inequality.
Bottom 84% of the sales just give 60% of the total purchases amount.""")

print("Which is the category with more sales?")

# Barplot of the categories
sns.countplot(data = sales_clean, x = "category")
plt.title("Number of Sales per Products Category", fontsize = 24, fontweight = "bold", fontstyle = "italic")
plt.xlabel(xlabel = "Category", fontsize = 18, fontweight = "bold")
plt.ylabel(ylabel = "", fontsize = 18, fontweight = "bold")
# Add some annotations
number_points = sales_clean.shape[0]        # number of points in the dataset
category_counts = sales_clean.category.value_counts() # all Values of the categories
plt.yticks([])
locs, labels = plt.xticks(fontsize=14)   # Current tick locations and labels
# Loop through each pair of locations and labels  
for loc, label in zip(locs, labels):
    # Text property for the label to get the current count
    count = category_counts[label.get_text()]
    percentage = '{:0.1f}%'.format(100*count/number_points) # Percentage of each category
    # Put the annotations inside the bar on the top
    plt.text(x = loc, y = count-10000, s = percentage, ha = 'center', color = 'white', fontsize=14)
sns.despine(left=True)
plt.show()
plt.savefig("images/sales_by_category.png");
print("The figure has been saved in images folder")
time.sleep(2)


print("62\% of the sales are in category 0, 33\% in category 1 and just 5.2\% in category 2.")
time.sleep(2) 

print("What is the Customer's Age with highest purchases")

print("Plot age distribution")
bins_sales_clean = int(round(1 + np.log2(sales_clean.shape[0])))# Optimal number of bins
sns.distplot(sales_clean.age, bins = bins_sales_clean) # Distribution plot
plt.xlabel("Age", fontsize = 18, fontweight = "bold")
plt.title("Distribution of Age", fontsize = 24, fontweight = "bold")
plt.axvline(sales_clean.age.mean(), color = "red")
plt.axvline(sales_clean.age.median(), color = "orange")
plt.legend(["Mean", "Median"], fontsize = 16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()
plt.savefig("images/hist_age.png");
print("The figure has been saved in images folder")
time.sleep(2)


print("Bivariate Analysis")

print("What is the Average price in each category?")

# Considering all the dataframe
# sns.catplot(x = "price", data = sales_clean, kind = "box", col = "category");

for i in range(3):
    print("Mean Category " + str(i) + ": ", sales_clean.price[sales_clean.category == str(i)].mean())
    print("Median Category " + str(i) + ": ", sales_clean.price[sales_clean.category == str(i)].median())
    print("Mode Category " + str(i) + ": ", sales_clean.price[sales_clean.category == str(i)].mode()[0])
    print("Standard Deviation Category " + str(i) + ":", sales_clean.price[sales_clean.category == str(i)].std(),"\n")    


print("""The average price in category `0 is $10, in category 1 it is $19 and in category 2 it is $62`. 
There are many outliers in all categories. we take as average the median.""")


# Average sales in each category removing outliers
# sns.catplot(x = "price", data = sales_clean_no_outliers, kind = "box", col = "category")
# # plt.show();

print("measures of center df without outliers:")
for i in range(3):
    print("Mean Category " + str(i) + ": ", sales_clean_no_outliers.price[sales_clean_no_outliers.category == str(i)].mean())
    print("Median Category " + str(i) + ": ", sales_clean_no_outliers.price[sales_clean_no_outliers.category == str(i)].median())
    print("Mode Category " + str(i) + ": ", sales_clean_no_outliers.price[sales_clean_no_outliers.category == str(i)].mode()[0])
    print("Standard Deviation Category " + str(i) + ":", sales_clean_no_outliers.price[sales_clean_no_outliers.category == str(i)].std(),"\n")


print("It seems that all the outiers values are from product of category 2. Product of category 2 are the most expensives.")
print("Most of the sales come from people aged between 30 and 55 with a peak at 42 years old.")
# 
print("Sales by Gender")

# Barplot of the categories
sns.countplot(data = sales_clean, x = "gender")
plt.title("Number of Sales by Gender", fontsize = 24, fontweight = "bold", fontstyle = "italic")
plt.xlabel(xlabel = "Gender", fontsize = 18, fontweight = "bold")
plt.ylabel(ylabel = "", fontsize = 18, fontweight = "bold")
# Add some annotations
number_points = sales_clean.shape[0]        # number of points in the dataset
category_counts = sales_clean.gender.value_counts() # all Values of each category
locs, labels = plt.xticks()   # Current tick locations and labels
# Loop through each pair of locations and labels  
for loc, label in zip(locs, labels):
    # Text property for the label to get the current count
    count = category_counts[label.get_text()]
    percentage = '{:0.1f}%'.format(100*count/number_points) # Percentage of each category
    # Put the annotations inside the bar on the top
    plt.text(x = loc, y = count-10000, s = percentage, ha = 'center', color = 'white', fontsize = 14)
sns.despine(left=True) # Remove the spine right and left
plt.xticks(fontsize=14)
plt.yticks(ticks=[], fontsize=14) # Remove all the ticks
plt.show()
plt.savefig("images/sales_by_gender.png");
print("The figure has been saved in images folder")
time.sleep(2)


print("The number of sales is almost the same for each gender.")

print("Sales over time")

# Check all the date availaible.

# Check the years 
sales_clean.date.dt.year.unique()

# Check the last date of 2022
sales_clean.date.max()

# We have data for 2021 and 2022. For 2022 we have just January and February.

print("Time series of the price each month") 
sales_over_time = sales_clean.copy()
sales_over_time["year"] = sales_over_time.date.dt.strftime("%Y") # Create year variable
sales_over_time["month"] = sales_over_time.date.dt.strftime("%b")# Create month variable
sales_over_time.month = pd.Categorical(sales_over_time.month, ordered=True, 
               categories=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
sales_over_time.groupby(["year", "month"]).price.sum().plot()# Plot sales over time
plt.title("Total Purchases Over Time", fontsize = 24, fontweight = "bold", fontstyle = "italic")
plt.xlabel("Date (2022 & 2021)", fontsize = 18, fontweight = "bold")
plt.ylabel("Purchases", fontsize = 18, fontweight = "bold")
plt.xticks(ticks=[2,3,4,5,6,7,8,9,10,11,12,13], 
           labels=["Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec","Jan", "Feb"], fontsize = 14)
plt.annotate(s="There is a huge drop in sales in October 2021",
             fontsize = 16,
             xy=(9,320000),
             xytext=(2,400000),
             arrowprops={"arrowstyle":"->", "color":"red"})
plt.yticks(fontsize=14)
plt.savefig("images/sales_over_time.svg");
print("The figure has been saved in images folder")
time.sleep(2)

print("Monthly growth rate")
# Compute the monthly sales, grouping by year and month, drop the missing values after grouping
monthly_sales = sales_over_time.groupby(["year","month"]).price.sum().reset_index().dropna()
# Compute the growth rate using the pct_change attribute, multiply by 100 to have the percentage
monthly_sales["growth_rate(%)"] = round(monthly_sales.price.pct_change()*100)
# Drop the price column
monthly_growth_rate = monthly_sales.drop(columns="price")
# Function to plot a custom table, Taken from stackoverflow 
#(https://stackoverflow.com/questions/26678467/export-a-pandas-dataframe-as-a-table-image)
def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    """This function will plot a custom table from a dataframe"""
    # Remove the plot figure as we just want the table
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')
    # Plot the table
    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)
    # Disable the automatic fontsize
    mpl_table.auto_set_font_size(False)
    # Set the fontsize
    mpl_table.set_fontsize(font_size)
    # Set the properties of the rows
    for k, cell in  six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    return ax
# Plot the table
render_mpl_table(monthly_growth_rate)
# Save the figure in the image folder
plt.savefig("images/monthly_growth_rate_table.svg")
print("The figure has been saved in the images folder")
average_growth_rate = monthly_growth_rate["growth_rate(%)"].mean()
print("On average we have a monthly growth rate of "+average_growth_rate+"\% even considering the huge drop ins sales of October\n.\
let's go deep to see what is really happening")

print("Sales in October 2021")

# Compute the dataframe with just october
sales_clean_oct = sales_clean[sales_clean.date.dt.month == 10]
# Group by category 
g_oct = sales_clean_oct.groupby("category").price.sum()
sns.barplot("category", "price", data = g_oct.reset_index(), palette = ["gray", "orange", "gray"])
plt.xlabel("Category", fontsize = 18, fontweight = "bold")
plt.ylabel("", fontsize = 18, fontweight = "bold")
plt.title("Purchases by Category in October 2021", fontsize = 24, fontweight = "bold", fontstyle = "italic")
# Put annotations on each bar

locs, labels = plt.xticks()
for loc, label in zip(locs, labels):
    count = g_oct[label.get_text()]
    sales = "${:,}".format(int(count))
    plt.text(x=loc, s=sales, y=count-10000, ha = "center", fontsize = 14, color = "white")
sns.despine(left=True)
plt.yticks([]) # Remove the ticks
plt.xticks(fontsize=14)
plt.show()
plt.savefig("images/october_category.png");
print("The figure has been saved in images folder")
time.sleep(2)


print("The Sales in October are very low for the category 1, let's go deep by checking each day of that month.")

# Dataframe of all the days of october, we group by category and days
sales_oct = sales_clean_oct.groupby(["category", sales_clean_oct.date.dt.day]).sum().reset_index()
# Filter just for category 1
sales_oct_1 = sales_oct[sales_oct.category == "1"]
sns.barplot("date", "price",data = sales_oct_1, color = "b")
plt.xlabel("Date", fontsize = 18, fontweight = "bold")
plt.ylabel("Purchases", fontsize = 18, fontweight = "bold")
plt.title("Category 1, Purchases in October", fontsize = 24, fontweight = "bold", fontstyle = "italic")
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()
plt.savefig("images/october_days.png");
print("The figure has been saved in images folder")
time.sleep(2)

print"We can clearly see that from the 2th to the 27th there were no sales. Maybe there was a lack of product that period, this is to investigate.")

print("What is the day of the week with the most sales over time?")
# Group by day and plot
sales_clean.groupby(sales_clean.date.dt.dayofweek).price.sum().plot(kind = "bar")
plt.xlabel("Day of Week (2021-2022)", fontsize = 18, fontweight = "bold")
plt.ylabel("Purchases", fontsize = 18, fontweight = "bold")
plt.title("Day of the Week Purchases", fontsize = 24, fontweight = "bold")
plt.xticks([0,1,2,3,4,5,6],["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], rotation = 0)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()
plt.savefig("images/days_of_week.png");
print("The figure has been saved in images folder")
time.sleep(2)


print("All the days of the week have almost the same amount of purchases, so there is not a better day for sales.\n\n")

print("Correlations\n")

print("Correlation between Gender and Category")

# boxplot, gender and categories of products purchased
# sns.catplot(x = "gender", y = "price", col = "category", kind = "box" , data = sales_clean)
# plt.show();

# We can see that in each category gender are almost simmetrical, this means that there is no difference of purchase between gender in each category.

# Our hypothesis.
# >   H0 : Gender and Category are independant  
#     H1: Gender and Category are dependant

# Function to find the dependence of 2 variables
def chi_squared(df, ind, col)-> str:
    """ Show if 2 variables of a dataframe are dependent or not"""
    # Contigency table
    contingency_table = pd.crosstab(index = df[ind], columns = df[col])
    stat, p, dof, expected = sci.chi2_contingency(contingency_table)
    if p > 0.01:
        return "{} and {} are probably independent".format(ind, col), contingency_table
    else:
        return "{} and {} are probably dependent".format(ind, col), contingency_table


# Find if the 2 variables are dependent
result, table = chi_squared(sales_clean, "category", "gender")

table.plot(kind="bar", rot= 0)
plt.title("Total Sales by Gender in Each Category", fontsize = 24, fontweight = "bold", fontstyle = "italic")
plt.xlabel("Category", fontsize = 18, fontweight = "bold")
plt.ylabel("Sales", fontsize = 18, fontweight = "bold")
plt.legend(fontsize = 18, loc = "upper right")
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()
plt.savefig("images/gender_by_category.png");
print("The figure has been saved in images folder")
time.sleep(2)

# Print the result of the test
print(result)
time.sleep(2)

print("""There are more sales in category 0 and very few in category 2, even if category sales 2 has products with higher prices.   
 In category **0** *Male* make more purchases than *Female*.  
 In category **1** *Female* make more purchases than *Male*.  
 In category **2** *Male* make more purchases than *Female*.""") 

print("Correlation between age and total amount of purchases")

# Create a dataframe with age and the total amount purchases
corr_age_price = sales_clean.groupby("age").sum().reset_index()
# Compute the pearsonr 
r_age_price = sci.pearsonr(corr_age_price.price, corr_age_price.age)[0]
# Plot using regplot
sns.regplot(x = "age", y = "price", data = corr_age_price)
plt.xlabel("Age", fontsize = 24, fontweight = "bold")
plt.ylabel("Total Amount of Purchases", fontsize = 24, fontweight = "bold")
plt.title("Total Amount of Purchases Vs. Age", fontsize = 24, fontweight = "bold")
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()
plt.text(s = "pearsonr = {:.2f}".format(r_age_price), y = 300000, x = 75, fontsize = 16);
print("The figure has been saved in images folder")
time.sleep(2)


print("""There is a strong negative correlation between the age and the amount of purchases, as the age increases the amount of purchases decreases.  
Young people buy more than old people.""")

print("Correlation between Age and the Purchase frequency(Number of purchases per month)")

# Number of purchases per month
age_vs_month = sales_clean.copy()
age_vs_month["month"] = age_vs_month.date.dt.month.astype("category") # Create the variable month in the dataframe
# Group by month then age to have the number of purchase each month for each age
age_vs_month_plt= age_vs_month.groupby(["month", "age"]).count().reset_index()
# Pearson r
r_age_month = sci.pearsonr(age_vs_month_plt.age, age_vs_month_plt.id_prod)[0]

# Plot
sns.regplot(x = "age", y = "id_prod", data = age_vs_month_plt)
plt.xlabel("Age", fontsize = 18, fontweight = "bold")
plt.ylabel("Number of Purchases Per Month", fontsize = 18, fontweight = "bold")
plt.title("Age vs. Number of Purchases per Month", fontsize = 24, fontweight = "bold", fontstyle = "italic")
# Put the pearsonr in the plot as a legend
plt.text(s = "pearsonr = {:.2f}".format(r_age_month), x = 80, y = 2600, fontsize = 16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()
plt.savefig("images/age_monthly_purchases.png");
print("The figure has been saved in images folder")
time.sleep(2)


print("""There is a correlation between the age and the number of purchases per month, as the age increases the number of purchases per month decreases.  
The number of purchases per month is higher for customers between 30 and 50 years old.""")

print("Correlation between Age and the Categories of purchases products")

print("""Our hypothesis:
     `H0 : Age and Category are independant`  
     `H1: Age and Category are dependant`""")


# Compute a group_age variable
def group_age(age:int) -> str:
    """return the group age of each age"""
    if age >= 18 and age < 35:
        return "18-34"
    elif age >= 35 and age < 61:
        return "35-60"
    elif age >= 61:
        return "61+"
# Create the group_age variable and turn it into category
sales_clean["group_age"] = sales_clean.age.apply(group_age).astype("category") 


# Use the function to find the dependence and contingency table
result_1, table_1 = chi_squared(sales_clean, "category", "group_age")
# Plot the heatmap
sns.heatmap(table_1, annot = True, cbar = False)
plt.title("Correlation Between Age Group & Category", fontsize = 24, fontweight = "bold", fontstyle = "italic")
plt.xlabel("Age Group", fontsize = 18, fontweight = "bold")
plt.ylabel("Category", fontsize = 18, fontweight = "bold")
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()
plt.savefig("images/heatmap_age_category.png");
print("The figure has been saved in images folder")
time.sleep(2)

# print the result of the test
print(result_1)
time.sleep(2)

# **Using ANOVA**
list_category = []
categories = sales_clean.category.unique()
for category in categories:
    list_category.append(sales_clean[sales_clean.category == category].age)
plt.boxplot(list_category, labels=categories, vert=False)
plt.show();
# Function to compute eta squared
def eta_squared(x,y):
    mean_y = y.mean()
    categories = []
    for category in x.unique():
        yi_category = y[x==category]
        categories.append({'ni': len(yi_category),
                        'mean_category': yi_category.mean()})
    TSS = sum([(yj-mean_y)**2 for yj in y])
    ESS = sum([c['ni']*(c['mean_category']-mean_y)**2 for c in categories])
    return ESS/TSS
eta_squared(sales_clean.category, sales_clean.age)


print("For all ages there are more sales in category 0 and less in category 2.  \n\
people aged between 35-60 make more purchases in category 0 and 1 compared to other age group.")

print("Correlation between age and the average basket size (in number of items)")
# compute the average basket size
age_average_basket = sales_clean.groupby(["age", "session_id"]).count().reset_index().groupby("age")["id_prod"].mean().reset_index()
sns.regplot("age", "id_prod", data = age_average_basket, ci = False)
plt.xlabel("Age", fontsize = 18, fontweight = "bold")
plt.ylabel("Average Basket Size", fontsize = 18, fontweight = "bold")
plt.title("Age vs. Average Basket Size", fontsize = 24, fontweight = "bold", fontstyle = "italic")
r_age_average_basket = sci.pearsonr(age_average_basket.age, age_average_basket.id_prod)[0]
plt.text(s = "pearsonr = {:.2f}".format(r_age_average_basket), x = 75, y = 2.45, fontsize = 16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()
plt.savefig("images/age_basket_size.png");
print("The figure has been saved in images folder")
time.sleep(2)

print("Using OLS(ordinary Leasts Squares)")

Y = age_average_basket.id_prod # Create a series with the average basket size 
X = age_average_basket[["age"]] # Create a dataframe with the age column
X = X.copy()  # Make a copy as the dataframe will be modify
X["intercept"] = 1. # Add the intercept column with value 1.
result = sm.OLS(Y, X).fit() # Fit the model
a,b = result.params["age"], result.params["intercept"] # variable a and b to plot the line ax+b

## Plot the regression line
plt.plot(age_average_basket.age, age_average_basket.id_prod, "^")
plt.plot(np.arange(93), [a*x+b for x in np.arange(93)])
plt.xlabel("Age")
plt.ylabel("Average Basket Size")
plt.show();

print("""There is a correlation between age and the average basket size.  
Young people (< 50 years old) have an average basket size higher than older people. As we could expect we have a pick
between people aged 30 to 50 years old which is the range where we have the highest sales.""")

time.sleep(5)
