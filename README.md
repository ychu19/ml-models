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
The dataset looks like this:
|    |          ean | name                                                                                                                                          |
|---:|-------------:|:----------------------------------------------------------------------------------------------------------------------------------------------|
|  0 | 725177540363 | Belle and The Yank 1/4" - 20 TPI x 90mm Hex Drive Flat Pancake Head Chrome Furniture Screw Bolt 3-35/64"                                      |
|  1 | 725177540370 | Belle and The Yank 1/4" - 20 TPI x 120mm Hex Drive Button Head Furniture Bolts Chrome Finish (4 Pack)                                         |
|  2 | 797776092321 | Eazel Wines 750ml Red Wine Eazel Shiraz 2014                                                                                                  |
|  3 | 701197194311 | IsyLei lilla IsyLei All-In-One, lilla - Cono LadyP in silicone medicale per la minzione femminile, con cappuccio salvagoccia, prolunga e spor |
|  4 | 797776110773 | MsConscious 370g Skinny Granola    

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

Also making every word in lower cases.

```python
pattern = '|'.join(['\d+', 'fl ', ' fl', 'oz ', ' oz', # removing numbers
                    ' cm ', ' g ', ' llc ', 
                    'inc ', ' inc', ' ct ',  ' lb ', 
                    ' ml ',  ' lbs ', ' pk ', # unit of capacity, etc. 
                   'limited', 'certified' , 'corporation', 'service', 'supermarkets])# other meaningless words
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
This is still not satisfying - `upc_food` still contains non-food items, like the Converse shoes:
|        |          ean | name                                         |
|-------:|-------------:|:---------------------------------------------|
| 868061 | 100740605005 | black computer rack screws tapered           |
| 868066 | 100847460019 | gm otb cr wildflower stick dsp               |
| 868073 | 101001010026 | bag apple                                    |
| 868094 | 100300345679 | rage against the machine                     |
| 868113 | 100013274488 | / authentic converse shoe                    |
| 868117 | 100036619655 | hes/,kg sneaky ferret / carpetshark          |
| 868119 | 100080466670 | sz usa men . converse chuck taylor all stars |
| 868124 | 100110021008 | wild girls fight club                        |
| 868131 | 100245134055 | well or loft wheel                           |
| 868133 | 100252003009 | game ticket                                  |

Therefore I removed irrelevant words such as Nike, Converse, and DKNY:

```python
irrelevant = ['pc', 'computer', 'dvd', 'vhs', 'laptop', 'game',
              'ipod','iphone','microsoft', 'software', # electronic items
              'video', 'media', 'cd', 'music', 'cards', 
              'record', 'entertainment',  # entertainment items
              'stainless', 'engine', 'screw', 'screws', 'bolt', 'machine',
              'mechanics', 'magnets', 'cable', 'wire', 'wheel', # hardware items
              'bible', 'pages', 'book', 'publishing', # books
              'dress', 'skirt', 'fleece', 'leggings', 
              'leather', 'jacket', 'xl','sweater', 'cashmere', # apparel
              'tablets', 'pharmaceuticals', 'acetaminophen', 'oxycodone', # medicine
              'nike', 'dkny', 'north face',  'oldnavy', 'converse', 
              'lauren ralph', 'wiley', 'mcgraw', 'walgreens', 'walgreen',# brands that don't sell food
              'comforter', 'plastic', 'cookware', 'luggage', 
              'cosmetics','shampoo', 'candle', 'pet'] # other irrelevant stuff
upc_food = upc_food[~upc_food.name.str.contains('|'.join(irrelevant))].reindex()
```
`.shape` tells us that we now have 64,228 food items in our dataset.

### Training Set and Test Sets

```python
upc_food_train = upc_food.sample(frac=0.6, random_state=0)
```
Dropping the training set to create test sets. Having three test sets to simulate three different grocery stores.
```python
upc_food_test = upc_food.drop(upc_food_train.index)
upc_food_test_1 = upc_food_test.sample(frac=0.25, random_state=0)
upc_food_test_2 = upc_food_test.sample(frac=0.15, random_state=0)
upc_food_test_3 = upc_food_test.sample(frac=0.35, random_state=0)
```

### Tokenizing Names

With `CountVectorizer` from `sklearn`, I converted the names to a matrix of token counts. 
```python
upc_food_1 = upc_food.sample(frac=0.1)

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(stop_words='english')
upc = count_vect.fit_transform(upc_food_1.name)
```

### Non-Negative Matrix Factorization (NMF)

```python
from sklearn.decomposition import NMF
nmf = NMF(n_components = 20, random_state=3,
          init='nndsvd').fit(upc_tf)
```
And here are the 20 topics we found :

Topics in NMF model (Frobenius norm):
+ Topic #0: oz la butter sauce ml ounces chicken net strawberry cut mix liquid body san soup
+ Topic #1: choice custom winner bowstrings tan green bowstring bowsting hoyt teardrop crossbow recurve bowtech spiral cc
+ Topic #2: tea great pacific atlantic ac ch lemon rice choice soda america mix peach beans pk
+ Topic #3: ahold tops fo beans dot se sb juice ct gnt peas sh grape red cut
+ Topic #4: chocolate ghirardelli milk dark chip bar white cookies squares gulliver cocoa cookie mint mix caramel
+ Topic #5: corp grinnell steel nipple galv bushing hex group revlon marketing trading evergreen block black joint
+ **Topic #6: foods rice lofthouse glory mix chicken spice nature pinnacle kraft gourmet natural chips simply asia**
+ Topic #7: green tea beans cut arizona sons rj ct ferolito vultaggio ml com ea bags bag
+ Topic #8: warner lambert consumer care health group ct adams halls listerine rolaids cherry drops benadryl spec
+ Topic #9: products kiss duckback rich home cake food blueberry consumer genisoy revlon natural superdeck division ross
+ Topic #10: lb red beef rice fresh pork white apples bag potatoes round veal yellow large ch
+ Topic #11: sh bruno ahold beef soup mix chicken bl rice chips bag vicks cmpn juice spaghetti
+ Topic #12: international orange glo fo juice group cleaner la ml drink vodka amigo jackel fuerza clean
+ Topic #13: black white red cherry pepper stripe large beans foxers small medium blue boyshort grinnell roll
+ Topic #14: apple juice pie sauce fo sour cider genuardi pk cinnamon cinn country cherry natural hill
+ Topic #15: giant ahold fo mix rice sauce beef pk grape tomato strawberry ct beans chicken juice
+ **Topic #16: cheese cream blue white cheddar pizza maytag sauce jack epi mac michelina bread genuardi italian**
+ Topic #17: dog food beef industries treats toy blackman chicken large bones biscuits cat group companion stores
+ Topic #18: bar choc clif butter chip cake peanut balance nt cookie cookies fo lemon cream pk
+ Topic #19: set piece kitchen tool mini enterprises kit pieces gift lego home baby international box puzzle

Among the 20 topics, I would say that Topic 6 is more or less about Asian food, and Topic 16 is about Italian food. 

## Predicting Food Categories in Test Sets

I use the `nmf` model above to predict food categories in the test sets.

```python
ddef rating_for_ethnic_food(test_data):
    """ 
    1. tokenize the name of each item in the dataset with fitted count_vect
    2. tf-idf transforming
    3. extract the topic number for each item
    4. combine the extracted topic number back to the test set
    5. calculate the proportion of ethnic food (topic 6 and topic 16) within the test set
    """
    test = count_vect.transform(test_data.name)
    tf_test = tf_transformer.transform(test)
    X_test = nmf_fit.transform(tf_test)
    predicted_topics = [np.argsort(each)[-1] for each in X_test]
    test_data['pred_topic_num'] = predicted_topics
    
    rating = round(
             (len(test_data[test_data.pred_topic_num==6]) + len(test_data[test_data.pred_topic_num==16])) 
              / len(test_data) * 100 , 2)
    print('Proportion of ethnic food in this supermarket is', rating, '%.')
```

```python
rating_for_ethnic_food(upc_food_test_1)
```
Proportion of ethnic food in this supermarket is 9.9 %.

```python
rating_for_ethnic_food(upc_food_test_2)
```
Proportion of ethnic food in this supermarket is 10.28 %.
```python
rating_for_ethnic_food(upc_food_test_3)
```
Proportion of ethnic food in this supermarket is 9.84 %.

Among the three hypothetical supermarkets, I would choose to visit the second one first. 

## Conclusion

The ultimate goal for this project is to create 
This document records the first step 

## Next Steps

