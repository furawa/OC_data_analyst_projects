#!/usr/bin/env python
# coding: utf-8

# # ANALYZE COMPANY'S SALES

# Load all the needed packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import os
import time

# Create a folder to save the file
os.makedirs("clean_datasets", exist_ok=True)
# Folder to save all the images
os.makedirs("images", exist_ok = True) # Create the folder to store all the images# get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_palette("colorblind")
sns.set_context("talk")
sns.set(style = "dark", rc={'figure.figsize':(11.7,8.27)})


print("Reading All Files..............................................................................")
time.sleep(2)
transactions = pd.read_csv("datasets/transactions.csv")
customers = pd.read_csv("datasets/customers.csv")
products = pd.read_csv("datasets/products.csv")

print("All the Datasets have been read")
time.sleep(2)

print("transactions:\n",transactions.head(2))
print("=======================================================================\n=======================================================================")
print(transactions.info())
print("=======================================================================\n=======================================================================")
print(customers.info())
print("=======================================================================\n=======================================================================")
time.sleep(5)
print("customers : \n", customers.head(2))
print("=======================================================================\n=======================================================================")
time.sleep(5)
print("products : \n", products.head(2))
print("=======================================================================\n=======================================================================")
time.sleep(5)
print(products.info())
print("=======================================================================\n=======================================================================")
time.sleep(5)

print("products stats : \n", products.describe())
print("=======================================================================\n=======================================================================")
time.sleep(5)
print("Cleaning All Datasets")
time.sleep(2)
print("Cleaning transactions dataset\n\n")
time.sleep(2)  
print("Define")
print("""
- **Some date starting with test must be split**
- **Date variable must be a *datetime* not a string**  
- **We must split id_prod in category and id_prod**
- **We must split session_id in session id and session_category**
- **We must split client_id in client_id and client_category**
- **Turn all the categories variable into a category data type** """)
time.sleep(10)
print("Code")
time.sleep(2)
# Write  a function to split a column
def split_columns(dataset, col):
    ### This function take a dataset and a column of the dataset split the column and return the 2 new columns
    new_col_1 = dataset[col].apply(lambda x : x.split("_")[1])
    new_col_2 = dataset[col].apply(lambda x : x.split("_")[0].upper())
    return new_col_1, new_col_2

# Create a copy of transactions dataset
transactions_df = transactions.copy()

# Split id_prod in 2 columns,id_prod and category
transactions_df["id_prod"], transactions_df["category"] = split_columns(transactions_df, "id_prod")

# Split client_id columns into 2 columns, client_id and client_category
transactions_df["client_id"], transactions_df["client_category"] = split_columns(transactions_df, "client_id")

# Split session_id in 2 columns, session_id and sesseion_category
transactions_df["session_id"], transactions_df["session_category"] = split_columns(transactions_df, "session_id")


# Check if everything is ok
transactions_df.head(2)

# Check the different categories
transactions_df.query("category == 'T'")


# There are 200 rows which date starts with test. We can guess that it was just to *test* if the system is working or not. These rows are not useful for our analysis. We will remove them.
# We can therefore notice that the test day was on 2021-03-01 at 02:30:02 am.

# Remove all the test dates
transactions_df = transactions_df.query("category != 'T'") # Select all the rows where the category is not T
transactions_clean = transactions_df.copy() # Create a new dataframe from transactions_df to avoid warnings



# Check if there are still test date, no output means there is no test date anymore
assert transactions_clean.category.all() != "T"

transactions_clean.date = transactions_clean.date.astype("datetime64")

# Assert that the date is in the correct type
transactions_clean.date.head()

# turn all the categories variable into a category data type
transactions_clean.iloc[:, 4:] = transactions_clean.iloc[:, 4:].astype("category")

print("cleaned transactions dataframe")
print("transactions cleaned :\n", transactions_clean.head(2))
print("=======================================================================\n=======================================================================")
print(transactions_clean.info())

print("=======================================================================\n=======================================================================")

print("Cleaning customers dataset..............................................")  
print("Define")  
print("""
- **Split client_id variable into 2 variables**
- **Turn sex variabble in uppercase**
- **Turn sex variable into category data type**

Code""")
time.sleep(10)
# Make a copy of customers dataset
customers_clean = customers.copy()

# Spllit client_id
customers_clean["client_id"], customers_clean["client_category"] = split_columns(customers_clean, "client_id")

# Turn sex in uppercase
customers_clean.sex = customers_clean.sex.map(lambda x : x.upper())

# Turn sex into category
customers_clean.sex = customers_clean.sex.astype("category")

print("Cleaned customers dataframe")
print("customer cleaned :\n", customers_clean.head())
print("=======================================================================\n=======================================================================")
print(customers_clean.info())
time.sleep(5)

print("Cleaning products dataset")  
print("""Define  
- **Categ variable must be category type not an int**
- **Change the categ name to category**  
- **Split the id_prod and keep just the id products**  
- **There is a price of -1, we will remove it**  
 
Code""")
time.sleep(10)
# Make a copy of the products dataset
products_df = products.copy()

# Change the categ name to category
products_df = products_df.rename(columns={"categ":"category"})

# Split id_prod and keep just the id products using the split_columns function
products_df["id_prod"], products_df["category"] = split_columns(products_df, "id_prod")

# Turn the category into a category type
products_df.category = products_df.category.astype("category")

products_df.head(2)
products_df.info()


# Check the -1 price row
products_df.query("price == -1")


# This is probably another test, we will remove it for a better analysis.

# In[35]:


products_clean = products_df.copy()
products_clean = products_clean.query("price != -1")


print("Some statistics from products dataframe after cleaning")
print("products cleaned stats : \n", products_clean.describe())
print("=======================================================================\n=======================================================================")

print("Now that all the datasets are clean we can join them all together in a unique dataset for anaylysis.")
time.sleep(2)
# Join all the datasets together
sales_merge = transactions_clean.merge(products_clean, on = ["category", "id_prod"], how = "left")

sales_df = sales_merge.merge(customers_clean, on = ["client_id", "client_category"], how = "left")


# We can see that price has less values than the other columns, probably due to missing values. Let us confirm that.

# Check the columns with missing values
sales_df.isna().any()

print("Plot the missing values in the datasets")
sales_df.isna().sum().plot(kind = "bar")
plt.show()
plt.savefig("images/missing_values.png")
print("Plot stored in the images folder");
time.sleep(2)
print("Check the id products with missing values.................................")
sales_df[sales_df.price.isna()]["id_prod"].unique()

print("There is just one product with no price.\nIt means that the information about this product **(id = 2245)** was not available in the product dataframe.")

print("We Replace the NaN values with the median price")
sales_df.price.fillna(sales_df.price.median(), inplace = True)
time.sleep(2)
# Assert that there are no missing values anymore in the dataframe
assert sales_df.isna().all().all() == False
print("=======================================================================\n=======================================================================")

print("Create the variable age, max date - birth date, and remove the birth column")
time.sleep(2)
actual_year = sales_df.date.max().year # Find the max year in the dataframe to use as actual year
sales_df["age"] = actual_year - sales_df.birth # Can lead to error if data is not updated over time
sales_df.head(2)

# Remove the useless column birth
sales_df = sales_df.drop("birth", axis = 1)
print(sales_df.head(2))
print("=======================================================================\n=======================================================================")
time.sleep(2)
print(sales_df.session_category.unique())
print(sales_df.client_category.unique())
print("client category and session category have just 1 category, those variables are not usefull for the analysis. We will remove them.")
time.sleep(2)
sales_clean = sales_df.drop(["client_category", "session_category"], axis = 1)
sales_clean.head(2)


print("Let us rename the **sex** variable to **gender**, and **F to female**, **M to male**, for better comprehension.")
time.sleep(2)
# Rename sex variable to gender
sales_clean.rename(columns = {"sex":"gender"}, inplace = True)
# Replace F by female and M by male
sales_clean.gender.replace("F", "Female", inplace = True)
sales_clean.gender.replace("M", "Male", inplace = True)
print("sales clean : \n", sales_clean.head())

print("=======================================================================\n=======================================================================")

print("Save all the Cleaned Datasets")
time.sleep(2)

# Save the final dataframe
sales_clean.to_csv("clean_datasets/sales_clean.csv", index=False)
# save the cleaned datasets
products_clean.to_csv("clean_datasets/products_clean.csv", index=False)
customers_clean.to_csv("clean_datasets/customers_clean.csv", index=False)
transactions_clean.to_csv("clean_datasets/transactions_clean.csv", index = False)
print("All the Datasets have been stored in the clean_datasets folder")
print("Now you can move to the Analysis part....................")
time.sleep(2)
# [Go to the next session. Analyze the Data.](analyze_sales.ipynb)
