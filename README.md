# Which Grocery Store Should I Go for East Asian Food? 
Classifying Food Items with Real-world Data

## Summary

Using grocery item data from scraping webpages at H Mart and Walmart, I built a Logistic Regression model that predicts whether an unknown item is East Asian (as in Korea, Japan, China, and Taiwan) or not. The purpose of this project is to help costumers to make a better decision about which grocery store to visit when looking for East Asian food. The model should be generalizable to other ethnic food items in the future.

## Table of Contents

## Motivation

I've been moving all over the United States for the past few years - Urbana-Champaign in IL, Charlottesville in VA, etc. Everytime I moved to a new place, I had to search for the right places for grocery. As a person with Korean background who grew up in Taipei, exlporing which supermarkets have the most East Asian food selection became a ritual for settling down in a new town. And it's never an easy task. 

I often found myself wandering between the aisles marked "international food" or "Asian food" at a Kroger or Harris Teeter. Of course, I can never find everything I need right away, and it's time consuming to shop around all the supermarkets just to find a good jar of gochujang. I realized that I was desparate when I ordered takeouts from a Korean restuarant for dishes I can totally make by myself just to ask the owner where to find good kimchi.

All the time, labor, and the desire to find food from home at American supermarkets motivate this project - the goal is to rate every supermarket regarding the amount of ethnic food provided. Customers can browse through the ratings to know which supermarket to visit beforehand. The rating should be a benefitial feature for both the consumers and the supermarkets - it saves time for consumers as they no longer need to go over all the supermarkets for the international food they want, and supermarkets with specialized products get to attract the customers who are actively looking for them. This project should also be generalizable to all different kinds of ethnic food, with East Asian food only as an example. 

## Goal of the Project: Identifying East Asian Food Items

Almost every grocery stores has *something* East Asian. The question is which ones have the most? If I could know which supermarket around me has the biggest collection of East Asian items, I could visit there first since I may have a better chance to find what I want. I wouldn't need to go to every supermarket randomly and hope to see what I want to see. It'll save me a lot of time. 

## Building a Logistic Regression Model

The model uses the names of food items as the feature and predicts whether the item is East Asian or not. I do so by scraping the food items from [H Mart](https://www.hmart.com/groceries) and [Walmart](https://www.walmart.com/browse/food/976759) to train a Logistic Regression model. The model will identify East Asian versus non-East Asian food among the food items from another supermarket. 

### Scraping Items from H Mart for East Asian Food with `BeautifulSoup`

H Mart is one of the most famous Korean grocery stores in the United States. I assume that all the items from the H Mart grocery page are East-Asian (N = 705).

<img src="https://github.com/ychu19/ml-models/blob/master/hmart_grocery.jpeg" width="900px" class="center">

```python
import requests
from bs4 import BeautifulSoup
from time import sleep # control the crawl rate to avoid hammering the servers with too many requests
from random import randint

def parsing_pages(list_of_pages):
    pages = []
    for page in list_of_pages:
        if pd.notnull(page):
            all_pages = requests.get(page)
            each_page = BeautifulSoup(all_pages.content, "html.parser")
            pages.append(each_page)
            sleep(randint(2,6))
        else: 
            pages.append(None)
    return pages

hmart = ['https://www.hmart.com/groceries?p=' + str(i) for i in range(1,16)]
hmart_parsed = parsing_pages(hmart)
```

```python
key = 'product name product-item-name'
items = [] 
for page in hmart_parsed:
    items_in_pages = page.find_all('strong', class_=key)
    for item in items_in_pages: # cleaning up the messy text
        items_per_page = item.get_text().replace('\n','').replace('\r','').strip().replace('     ', ' ').lower()
        items.append(items_per_page)

# extracting item names from the text
brands = [i.split('        ')[0] for i in items]
item_names = [i.split('        ')[-1] for i in items]

# extracting volumns at the end
regrex = re.compile(r'\d\..*')
item_volumns = [regrex.findall(i) for i in item_names]

## asian items with brand names
asian_items = []
for i, j in zip(brands, item_names):
    asian_item = i + ' ' + j
    asian_items.append(asian_item)

# getting rid of texts about volumns 
asian_items_no_volumns = []
for i in asian_items:
    asian_item = re.sub(' \d\..*', '', i)
    asian_items_no_volumns.append(asian_item)

columns = {
    'item_name': asian_items_no_volumns,
    'item_volumn': item_volumns,
    'type': 'asian'
}
asian_items = pd.DataFrame(data = columns)
```
The data looks like this:
|    | item_name                              | item_volumn              | type   |
|---:|:---------------------------------------|:-------------------------|:-------|
|  0 | cj hetbahn cooked white rice box       | ['7.4oz(210g) 12 ea']    | asian  |
|  1 | cj cooked white rice with seaweed soup | ['5.9oz(167g)']          | asian  |
|  2 | samyang hot chicken flavor ramen       | ['4.94oz(140g) 5 packs'] | asian  |
|  3 | haioreum peeled roasted chestnuts      | ['3.52oz(100g)']         | asian  |
|  4 | paldo bibim men                        | ['4.58oz(130g) 5 packs'] | asian  |


### Scraping Items from Walmart for Non-East Asian Food with `selenium`

Walmart conveniently has separate categoies for Asian food, and therefore I assume that all the other grocery items are non-Asian. I use `selenium` in Python for the dynamic web pages on Walmart.com. 

<img src="https://github.com/ychu19/ml-models/blob/master/walmart_grocery.jpeg" width="900px" class="center">

```python
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import chromedriver_binary # adds chromedriver binary to path

driver = webdriver.Chrome()
driver.get("https://www.walmart.com/browse/food/")

list_of_items_from_pages = []
for n in range(1,26):
        driver.get("https://www.walmart.com/browse/food/?page=" + str(n))
        items_from_pages = driver.find_elements_by_css_selector(".product-title-link.line-clamp span")
        for i in items_from_pages:
            items_extracted = i.text
            list_of_items_from_pages.append(items_extracted)
        sleep(3)

        
# getting rid of units/volumns at the beginning of each item
pack = re.compile(r'\(\d{1,} (?:pack|count|cans)\)') 
list_of_items_from_pages = [re.sub(pack, '', i.lower()) for i in list_of_items_from_pages]

# isolating volumns at the end
regrex = re.compile(r'(?:, \d{1,}|\d{1,} (?:mg|oz|fl|ct)|, \d{1,} (?:mg|oz|fl|ct)).*')
walmart_item_volumns = [regrex.findall(i) for i in list_of_items_from_pages]
walmart_item_names = [re.sub(regrex, '', i.split(',')[0].strip()) for i in list_of_items_from_pages]

columns = {
    'item_name': walmart_item_names,
    'item_volumn': walmart_item_volumns,
    'type': 'non-asian'
}
walmart_items = pd.DataFrame(data = columns)
```

The data looks like this:
|    | item_name                                             | item_volumn                   | type      |
|---:|:------------------------------------------------------|:------------------------------|:----------|
|  0 | doritos nacho cheese flavored tortilla chips          | [', 15 oz']                   | non-asian |
|  1 | luvs triple leakguards extra absorbent diapers size 4 | ['88 ct']                     | non-asian |
|  2 | high-quality 4 x 6 prints                             | []                            | non-asian |
|  3 | equate 70% isopropyl alcohol                          | [', 32 oz']                   | non-asian |
|  4 | lipton diet green tea citrus iced tea                 | [', 16.9 fl oz (24 bottles)'] | non-asian |

### Pre-processing the data

First, I concat the two dataframes together 
```python
frames = [walmart_items, asian_items]
df = pd.concat(frames).reset_index(drop = True)
```

And then I separate training and test sets
```python
from random import sample
df_train = df.sample(frac=0.6, random_state=0)
df_test = df.drop(df_train.index)
```

#### Tokenizing/vectorizing the Item Names
```python
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfTransformer

count_vect = CountVectorizer(stop_words='english',max_df=0.85)
train_vect = count_vect.fit_transform(df_train.item_name)

# training set
train_tf_transformer = TfidfTransformer(use_idf=False).fit(train_vect)
train_tf = train_tf_transformer.transform(train_vect)
train_tf.shape

# test set
test_vect = count_vect.transform(df_test.item_name) # not fitting, but transforming
test_tf = train_tf_transformer.transform(test_vect)
test_tf.shape
```

### Building the Model

```python
from sklearn.linear_model import LogisticRegression

X_train = train_tf
y_train = df_train.type

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

X_test = test_tf
y_test = df_test.type

log_reg.score(X_test, y_test)
```
The model has a score of `0.9337` - not too bad! 

## Use the Model to Categorize East Asian Items at Wegmans and Harris Teeter

I scraped items at the international food sections of Wegmans and the Asian food sections at Harris Teeter, and use the model to categorize East Asian food from all the other international food.

### Wegmans

<img src="https://github.com/ychu19/ml-models/blob/master/wegmans_grocery.jpeg" width="900px" class="center">

#### Scraping Items at Wegmans' International Food Selection

```python
wegmans_international_items = []
driver.get("https://shop.wegmans.com/shop/categories/345?page=1")
sleep(3)
driver.find_element_by_css_selector("#shopping-selector-shop-context-intent-instore").click()
sleep(3)
for n in range(1,26):
    sleep(2)
    driver.get("https://shop.wegmans.com/shop/categories/345?page=" + str(n))
    sleep(3)
    items = driver.find_elements_by_css_selector(".cell-title")
    for i in items:
        items_extracted = i.text
        wegmans_international_items.append(items_extracted)
    sleep(randint(3,10))

wegmans_intl_item_names = [i.split('\n')[0] for i in wegmans_international_items]
wegmans_intl_item_volumns = [i.split('\n')[1] for i in wegmans_international_items]

columns = {
    'item_name': wegmans_intl_item_names,
    'item_volumn': wegmans_intl_item_volumns
}
wegmans_intl = pd.DataFrame(data = columns)
```

#### Use the Logistic Regression Model on the International Food Items

First tokenize the words in item names:
```python
wegmans_intl_vect = count_vect.transform(wegmans_intl.item_name)
wegmans_intl_tf = train_tf_transformer.transform(wegmans_intl_vect)
```
And then apply the model:
```python
X_intl = wegmans_intl_tf
y_intl = log_reg.predict(X_intl)
```
Here are some of the East Asian food my model found:
|    | item_name                                     | item_volumn   | type   |
|---:|:----------------------------------------------|:--------------|:-------|
| 10 | Wegmans Organic Salsa, Medium                 | 15.5 ounce    | asian  |
| 12 | Wegmans Organic Mild Salsa                    | 15.5 ounce    | asian  |
| 19 | Kikkoman Soy Sauce, Less Sodium               | 10 fl. oz.    | asian  |
| 26 | Huy Fong Chili Sauce, Hot, Sriracha           | 17 ounce      | asian  |
| 27 | Wegmans Reduced Sodium Soy Sauce              | 11.4 fl. oz.  | asian  |
| 28 | Wegmans Organic Sesame Garlic Sauce           | 14 ounce      | asian  |
| 35 | Wegmans Organic Teriyaki Sauce                | 14 ounce      | asian  |
| 36 | Old El Paso Enchilada Sauce, Red, Mild        | 10 ounce      | asian  |
| 40 | Wegmans Organic Black Bean Dip                | 16.5 ounce    | asian  |
| 42 | Wegmans Diced Green Chile Peppers             | 7 ounce       | asian  |
| 43 | Wegmans Whole Chipotle Peppers in Adobo Sauce | 7 ounce       | asian  |
| 45 | Kikkoman Soy Sauce                            | 10 fl. oz.    | asian  |

Not the most perfect result - "Salsa" and "Enchilada Sauce" are definitely NOT East Asian. Nonetheless, I'm pretty happy with most of the items identified by the model. 

### Harris Teeter

Harris Teeter happens to have a category called "Asian Food"! But how many items are East Asian? To scrape the items from HT, I'm using a different approach because it has different pages dedicated to different sub-categories, such as "Noodles" and "Sauces", as shown here:
<img src="https://github.com/ychu19/ml-models/blob/master/harris_teeter_grocery.jpeg" width="900px" class="center">

#### Scraping the Pages
I wrote a function to scrape all the pages across different sub-categories:
```python
def scraping_ht_pages(url, last_page_number):
    list_of_items_ht = []
    if last_page_number == 1:
        driver.get(url)
        sleep(randint(2,6))
        elements = driver.find_elements_by_css_selector(".product-name")
        for i in elements:
            list_of_items_ht.append(i.text)
        sleep(randint(2,6))
        return list_of_items_ht
    else:
        for n in range(1, last_page_number+1):
            driver.get(url + str(n) + "&appliedSort=Brand")
            sleep(randint(2,6))
            elements = driver.find_elements_by_css_selector(".product-name")
            for i in elements:
                list_of_items_ht.append(i.text)
            sleep(randint(2,6))
        return list_of_items_ht
```
And use the function to scrape all the pages:
```python
## different pages of Asian food
canned_url = "https://www.harristeeter.com/shop/store/332/category/583/subCategory/713,1557/products?isSpecialSubCategory=false"
ht_canned = scraping_ht_pages(canned_url, 1)

noodles_url = "https://www.harristeeter.com/shop/store/332/category/583/subCategory/713,976/products?pageNo="
ht_noodles = scraping_ht_pages(noodles_url, 2)

others_url = "https://www.harristeeter.com/shop/store/332/category/583/subCategory/713,1558/products?isSpecialSubCategory=false"
ht_other = scraping_ht_pages(others_url, 1)

sauces_url = "https://www.harristeeter.com/shop/store/332/category/583/subCategory/713,912/products?pageNo="
ht_sauces = scraping_ht_pages(sauces_url, 3)

seasoning_url = "https://www.harristeeter.com/shop/store/332/category/583/subCategory/713,1231/products?isSpecialSubCategory=false"
ht_seasoning = scraping_ht_pages(seasoning_url, 1)

ramen_url = "https://www.harristeeter.com/shop/store/332/category/583/subCategory/713,1164/products?pageNo="
ht_ramen = scraping_ht_pages(ramen_url, 2)

harris_teeter_asian = ht_canned + ht_noodles + ht_other + ht_sauces + ht_seasoning + ht_ramen
ht_asian = pd.Series(harris_teeter_asian)
```
#### Tokenizing the Item Names

```python
ht_asian_vect = count_vect.transform(harris_teeter_asian)
ht_asian_tf = train_tf_transformer.transform(ht_asian_vect)
```
#### Apply the Model

```python
X_ht = ht_asian_tf
y_ht = log_reg.predict(X_ht)
```
Here's a list of the East Asian food my model identified:
|    | item_name                                             | type   |
|---:|:------------------------------------------------------|:-------|
|  7 | Sharwood's Curry Cooking                              | asian  |
|  9 | Annie Chuns Maifun Brown Rice Noodles                 | asian  |
| 10 | Annie Chuns Noodle Soup - Miso with Tofu and Scallion | asian  |
| 16 | Hakubaku Organic Ramen Noodles                        | asian  |
| 17 | KA ME Rice Sticks                                     | asian  |
| 19 | Ka-Me Pad Thai Express Rice Noodles                   | asian  |
| 22 | Ka-Me Thai Rice Noodles - Stir Fry                    | asian  |
| 24 | La Choy Rice Noodles                                  | asian  |
| 25 | Maruchan Ramen Noodle Soup, Gold, Spicy Miso Flavor   | asian  |
| 28 | Tasty Bite Rice, Organic, Smoky Chipotle              | asian  |

## So Which One Should I Visit First: Wegmans or Harris Teeter?

Comparing the proportion of East Asian food items at both places:
```python
rate = round(len(y_intl[y_intl == 'asian']) / len(y_intl), 2) # for Wegmans
rate_ht = round(len(y_ht[y_ht == 'asian']) / len(y_ht), 2) # for Harris Teeter
```
Wegmans has 37% of East Asian food while HT has 65%. It seems like I should visit HT first. 
One caveat is that, when scraping the data, I had the whole international food department from Wegmans, but only the Asian food department from HT. So let's check the *number* of East Asian food items at both places:

```python
print(The number of East Asian food items at Wegmans:", len(y_intl[y_intl == 'asian']))
print(The number of East Asian food items at Harris Teeter:", len(y_ht[y_ht == 'asian']))
```
The number of East Asian food items at Wegmans: 133
The number of East Asian food items at Harris Teeter: 88

I would probably still choose Wegmans for the size of their East Asian Food selection!
