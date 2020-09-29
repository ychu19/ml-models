# Identifying Food Categories with Unsupervised ML \& NLP

I've been moving all over the United States for the past few years - Urbana-Champaign in IL, Los Angeles in CA, and Charlottesville in VA. Everytime I moved to a new place, I had to search for the right places for grocery. As a person with Korean background who grew up in Taipei, exlporing which supermarkets have the most Asian food selection became a ritual for settling down in a new town. And it's never an easy task (well, not so hard in LA, but that's LA!). 

In small college towns (and perhaps most of the places in the U.S.), it is difficult to find a nice Asian market. I often found myself wandering between the aisles marked "international food" or "Asian food" at a Kroger or Harris Teeter. Of course, I can never find everything I need right away, and it's time consuming to shop around all the supermarkets just to find a good jar of gochujang. I realized that I was desparate when I ordered takeouts from a Korean restuarant for dishes I can totally make by myself just to ask the owner where to find good kimchi. 

All the time, labor, and the desire to find ethnic food at American supermarkets motivate this project - the goal is to rate every supermarket regarding the amount of ethnic food provided. Customers can browse through the ratings to know which supermarket to visit beforehand. The rating should be a benefitial feature for both the consumers and the supermarkets - it saves time for consumers as they no longer need to go over all the supermarkets for international food, and supermarkets with specialized products get to attract the customers who are actively looking for them. 

## Goal of the Project

The ultimate goal is to create a rating for each supermarket around the users regarding the amount of specific ethnic food (like Ethiopian, Chinese, Korean food). This readme.md demonstrates the first step of the project - using Non-Negative Matrix Factorization (NMF) to identify ethnic food items based on the item titles, and create a measure/rating for imaginary supermarkets regarding the amount of ethnic food they have. 

## Identifying Ethnic Food from [UPC Barcode Data on Kaggle](https://www.kaggle.com/rtatman/universal-product-code-database)

According to [Wikipedia](https://en.wikipedia.org/wiki/Universal_Product_Code), "The Universal Product Code (UPC; redundantly: UPC code) is a barcode symbology that is widely used in the United States, Canada, Europe, Australia, New Zealand, and other countries for tracking trade items in stores." The UPC "consists of 12 numeric digits that are uniquely assigned to each trade item."

### Pre-processing the Data

First I imported the `csv` data, read it as a table, and make the columns as strings:

```python
upc = pd.read_table('upc_corpus.csv', delimiter=',' , dtype={"ean": np.str , "name": np.str})
```

`.shape` tells us that there are 1,048,575 examples from the `upc` data. 

#### Removing the nulls by excluding `len(ean)!=12` 

```python
upc = upc[upc.ean.str.len()==12]
```

#### Removing the nulls by excluding examples w/o names

```python
upc = upc[upc.name.notnull()]
```

#### Removing meaningless words in the names

```python
pattern = '|'.join(['\d+', 'fl ', ' fl', 'oz ', ' oz', # removing numbers
                    ' cm ', ' g ', 'llc ', ' llc', 
                    'inc ', ' inc', ' ct ',  ' lb ', 
                    ' ml ',  ' lbs ', ' pk ', # unit of capacity, etc. 
                   'limited', 'certified' , 'corporation', 'service'])# other meaningless words
upc['name'] = upc.name.str.lower().str.replace(pattern, '') 
```

### Identifying Food Items from the `upc` Data

The `upc` data contains not only food items but many other things, such as computers, apparel, etc. We need to isolate food items first. 

```python
from nltk.corpus import wordnet as wn
food = wn.synset('food.n.02')
food = list(set([w for s in food.closure(lambda s:s.hyponyms()) for w in s.lemma_names()])) # as reference
food = [sub.replace('_', ' ') for sub in food] # to match upc.name's format 
```
But `food` from `wordnet` doesn't really contain names of ethnic food. Here I manually added a few to the `food` list.

```python
ethnic_food = ['rice', 'ramen', 'sesame oil', 'curry', 'mirin', 'miso', 'nori', 'teriyaki', 'sriracha', 'soy']
food.extend(ethnic_food)
```
And select the examples that contain any words from `food` to create a list of food items from `upc`.

```python
upc_food = upc[upc.name.str.contains('|'.join(food))].reindex()
```
This is still not satisfying - 
