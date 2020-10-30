"""
Description:
    Code regarding assignment 1 of the slimstock assessment

Author: 
    Kevin Overbeek

Date:
    2020/10/16
"""

# In[Imports]
# NOTE: I'm not familiar with this, what does this do/what is this for?

import os

import numpy as np
import pandas as pd

# In[Product analysis class]
# NOTE: Nice that you demonstrate the use of classes. In general however,
# you should think about if what you're doing is more fit for a class of
# function.
class ProductAnalysis():
    
    
    """
    Class to load the category, product and transaction datasets, derive 
    the child-parent relationships and calculate some basic quantities for 
    each of the top-level categories
    
    Assumes the presence of the following files:
        - categories.csv
        - products.csv
        - transactions.csv
    """
    # NOTE: I like the docstrings.
    
    def __init__(self, datapath):
        # NOTE: The usual convention is to use names like "data_path".
        
        """
        Initialize the product analysis class. Checks if all the required
        files are present before reading them and storing them in the class.
        
        Parameters
        ----------
        
        datapath: string
            Path where the datasets are stored
        """
        
        # Store datapath
        self.datapath = datapath
        
        # NOTE: Nice that you check this here.
        # Check if expected files are present
        assert 'categories.csv' in os.listdir(), "No file present with the \
        name 'categories.csv' in passed datapath"

        assert 'products.csv' in os.listdir(), "No file present with the \
        name 'products.csv' in passed datapath"

        assert 'transactions.csv' in os.listdir(), "No file present with the \
        name 'transactions.csv' in passed datapath"
        
        # Load datasets and store in class
        self.categories = pd.read_csv('categories.csv')
        self.products = pd.read_csv('products.csv')
        self.transactions = pd.read_csv('transactions.csv')
        # NOTE: You can wonder whether it is a good idea to attach the
        # actual (potentially large) data to the class. You would like to avoid
        # making copies too often. It probably makes more sense to put the logic
        # in the class and provide the data is inputs (or put everything in
        # functions instead). However, these considerations go a bit far for
        # this assignment.

    def extract_top_categories(self):
        
        """
        Method to filter out the highest level of categories. Does this by
        checking which of the parent categories do not have a parent of
        their own, which makes them a top-level category
        """
        
        # Extract column with parent categories, filter to keep unique values
        parent_categories = self.categories['Parent Category']
        parent_categories = pd.Series(parent_categories.unique())
        
        # Extract column with categories
        categories = self.categories['Category']

        # Extract parent categories that do not have a paranet category of
        # their own (i.e. they do not exist in the category column)
        top_categories = parent_categories[~parent_categories.isin(categories)]
        self.top_categories = top_categories

        # Print outcome
        print("\n The top-level categories are: \n%s" % (top_categories))

        return

    def match_top_categories(self):
        
        """
        Method to link each category to the correct top-level category. Does 
        this by working its way down level by level until there are no more
        categories without the correct top-level category.
        """
        
        # Extract categories dataset, add empty column for 'Top Category'
        categories = self.categories.copy()
        categories['Top Category'] = np.nan
        
        # For all categories whose parent is a top category, copy this 
        # parent category
        categories.loc[categories['Parent Category'].isin(self.top_categories), 
                       'Top Category'] = categories['Parent Category']

        # While we have NA values in the 'Top Category' column, repeat:
        NA_count = categories['Top Category'].isna().sum()
        while NA_count > 0:
    
            # Merge categories on their parent
            categories_merged  = categories.merge(categories, 
                                                  how='left', 
                                                  left_on = 'Parent Category', 
                                                  right_on = 'Category', 
                                                  suffixes = ('', '_parent'), 
                                                  validate = 'many_to_one')
            # NOTE: The usual convention is that keyword arguments do not get spaces.
            # The following is more subjective, but I prefer to organize long lines as:
            '''
            categories_merged = categories.merge(
                categories,
                how='left',
                left_on='Parent Category',
                right_on='Category',
                suffixes=('', '_parent'),
                validate='many_to_one',
            )
            '''
            
            # For all rows with NA as 'Top Category', replace NA by the 'Top
            # Category' of the parent (which can also be NA)
            categories['Top Category'].fillna(categories_merged['Top Category_parent'], 
                                              inplace = True)

            # NOTE: Glad to see this here. You could add a warning or raise an error also.
            # Break if we do not fill new rows to avoid infinite loops
            NA_count_new = categories['Top Category'].isna().sum()
            if NA_count == NA_count_new:
                break
            else:
                NA_count = NA_count_new

        # Place new categories dataframe in class
        self.categories = categories

        return
    
    def analyze_transactions(self):
        
        """
        Linkes the categories dataset (with their respective top-level 
        category) to the products and transactions datasets. Also calculates
        the required number of products, average price and total value
        """
        
        # Merge products with categories dataset
        self.products = self.products.merge(self.categories, 
                                            how='left', 
                                            on = 'Category',
                                            validate = 'many_to_one')
        
        # Group products by 'Top Category'
        grouped_products = self.products.groupby('Top Category')
        
        # Merge transactions with products dataset, add value column
        self.transactions = self.transactions.merge(self.products, 
                                                    how='left', 
                                                    on = 'Product ID',
                                                    validate = 'many_to_one')
        self.transactions['Value'] = self.transactions['Quantity'] * self.transactions['Price']
        
        # Group transactions by 'Top Category'
        grouped_transactions = self.transactions.groupby('Top Category')
        
        # Create output dataframe, add desired columns one by one
        output = pd.DataFrame(index = self.top_categories)
        output['Category'] = output.index
        output['Number of Products'] = self.products['Top Category'].value_counts()
        output['Average Price'] = grouped_products.mean().round(2)
        output['Total Value'] = grouped_transactions.sum()['Value'].round(2)
    
        # Verify output
        # NOTE: Again very nice that you do this!
        assert output['Number of Products'].sum() == self.products.shape[0], \
        'Not all products are allocated to a top-level category'
    
        assert output['Total Value'].sum() == self.transactions['Value'].sum(), \
        'Sum of all category values does not add up to total value'
    
        # Store output in class
        self.output = output
    
    def write_output(self, outputfolder):
        
        """
        Write output to the desired folder in a csv format, leaves out index
        since this column overlaps with the 'Category' column
        
        Parameters
        ----------
        
        outputfolder: string
            Path where the output is stored
        
        """
        
        # Print output
        print("\n Output of product analysis: \n")
        print(self.output)
        
        # Write output to folder
        self.output.to_csv(outputfolder + 'output.csv', index=False)
        
        return

# In[Execution of class]

# NOTE: It's useful to use the following, then you could
# import your code as a module without running the code.
# if __name__ == "__main__":
#     pass

# NOTE: To avoid hard-coded directory paths, you could use something like
# a configuration file (that is not checked in in the repository).
# Set datapath as current directory
datapath = 'D:/Kevin Overbeek/Documents/Slimstock assessment/'
os.chdir(datapath)

# Create class, call all methods in the correct order
Analysis = ProductAnalysis(datapath)
Analysis.extract_top_categories()
Analysis.match_top_categories()
Analysis.analyze_transactions()
Analysis.write_output(datapath)