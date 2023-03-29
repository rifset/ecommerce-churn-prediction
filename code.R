# generic
library(data.table)
library(tidyverse)
library(lubridate)

# visualization
library(ggthemr)
library(ggrepel)
library(scales)
library(magick)

# text preprocessing / NLP
library(tm)
library(SnowballC)

# classification
library(caTools)
library(caret)
library(randomForest)


# data preparation --------------------------------------------------------

trx_data <- fread("Sales Transaction.csv")
trx_data[, Date := mdy(Date)]
glimpse(trx_data)


# data preprocessing ------------------------------------------------------

user_fm <- trx_data %>% 
  filter(Quantity > 0) %>% 
  group_by(CustomerNo) %>% 
  summarize(
    PurchaseFrequency = uniqueN(TransactionNo),
    TotalValue = sum(Price*Quantity)
  )

user_retention <- trx_data %>% 
  filter(Quantity > 0) %>% 
  group_by(CustomerNo) %>% 
  arrange(Date) %>% 
  summarize(
    FirstTrx = first(Date),
    LastTrx = last(Date)
  ) %>% 
  left_join(
    trx_data %>% 
      filter(Quantity > 0) %>% 
      group_by(CustomerNo) %>% 
      mutate(DateRank = frankv(Date, ties.method = "dense", order = -1)) %>% 
      filter(DateRank == 2) %>% 
      group_by(CustomerNo) %>% 
      summarize(SecondLastTrx = unique(Date)),
    by = "CustomerNo"
  ) %>% 
  mutate(
    SecondLastTrx = if_else(is.na(SecondLastTrx), FirstTrx, SecondLastTrx),
    LastObservedDate = max(trx_data$Date)
  ) %>% 
  relocate(LastTrx, .after = "SecondLastTrx") %>% 
  mutate(
    SecondLastDiff = LastTrx - SecondLastTrx,
    FirstLastObsDiff = LastObservedDate - FirstTrx
  )

user_churn <- user_fm %>% 
  left_join(user_retention, by = "CustomerNo") %>% 
  mutate(
    NewUser = if_else(FirstLastObsDiff <= 90, "yes", "no"),
    Churn = if_else((PurchaseFrequency == 1 & FirstLastObsDiff > 90) | 
                      (PurchaseFrequency > 1 & SecondLastDiff > 90), "yes", "no")
  )

user_bought <- trx_data %>% 
  filter(Quantity > 0) %>% 
  group_by(CustomerNo) %>% 
  summarize(ProductBought = str_flatten(ProductName, "; "))

# data visualization ------------------------------------------------------

ggthemr(
  palette = "flat",
  spacing = 1.15
)
myTheme1 <- theme(
  plot.title = element_text(size = 18),
  plot.title.position = "plot",
  plot.caption = element_text(hjust = 0, face = "italic", size = 8),
  plot.caption.position = "plot"
)

trx_monthly_stat <- trx_data %>% 
  filter(Quantity > 0) %>% 
  group_by(Month = floor_date(Date, "month")) %>% 
  summarize(
    TrxCount = uniqueN(TransactionNo),
    UniqueCust = uniqueN(CustomerNo)
  )
trx_monthly_stat

trx_monthly_stat %>% 
  select(Month, TrxCount, UniqueCust) %>% 
  pivot_longer(cols = -Month) %>% 
  ggplot(aes(x = Month, y = value, color = name)) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 3) +
  geom_label_repel(aes(label = value), fontface = "bold", show.legend = FALSE) +
  scale_color_discrete(labels = c("Transaction Count", "Unique Customer")) +
  labs(
    title = "Monthly Transaction Stats"
  ) +
  myTheme1 +
  theme(
    axis.title = element_blank(),
    legend.title = element_blank(),
    legend.position = "bottom"
  )

trx_regional_stat <- trx_data %>% 
  filter(Quantity > 0) %>% 
  group_by(Country) %>% 
  summarize(
    TrxCount = uniqueN(TransactionNo),
    UniqueCust = uniqueN(CustomerNo),
    GMV = sum(Price*Quantity)
  )
trx_regional_stat %>% 
  arrange(desc(GMV))

trx_regional_stat %>% 
  mutate(Country = fct_lump_n(Country, n = 1, w = GMV)) %>% 
  group_by(Country) %>% 
  summarize(GMV = sum(GMV)) %>% 
  mutate(`% GMV` = percent(GMV/sum(GMV), accuracy = .1)) %>% 
  ggplot(aes(x = Country, y = GMV, fill = Country)) +
  geom_bar(stat = "identity", width = .75, show.legend = FALSE) +
  geom_text(aes(label = `% GMV`, color = Country), 
            fontface = "bold", vjust = -.5, show.legend = FALSE) +
  scale_y_continuous(
    labels = label_dollar(scale = 1e-6, prefix = "£", suffix = "M", big.mark = ""),
    expand = expansion(c(0, .1), 0)
  ) +
  labs(
    title = "GMV Distribution by Buyer Country",
  ) +
  myTheme1 +
  theme(
    axis.title.x = element_blank()
  )

most_bought_product <- trx_data %>% 
  filter(Quantity > 0) %>% 
  count(ProductName, sort = TRUE)
most_bought_product

most_bought_product %>% 
  slice_max(order_by = n, n = 10) %>% 
  ggplot(aes(x = fct_reorder(ProductName, n), y = n)) +
  geom_bar(stat = "identity", width = .75) +
  scale_x_discrete(label = wrap_format(21)) +
  coord_flip(ylim = c(1000, 2500)) +
  labs(
    title = "Most Bought Product",
    x = "Product Name",
    y = "Sold Count"
  ) +
  myTheme1 +
  ggplot2::annotate(
    geom = "rect", xmin = 1.3, xmax = 5.2, ymin = 1880, ymax = 2320,
    fill = "white"
  ) +
  annotation_raster(
    as.raster(image_fill(image_read("cream-hanging-heart-t-light-holder.jpg"), "none")),
    xmin = 2, xmax = 5, ymin = 1900, ymax = 2300
  ) +
  ggplot2::annotate(
    geom = "text", y = 2100, x = 1.65,
    label = "Cream Hanging Heart T-Light Holder\n(source: www.mochaberry.co.uk)",
    fontface = "bold"
  )

# text preprocessing ------------------------------------------------------

corpus_data <- Corpus(VectorSource(user_bought$ProductBought))
corpus_data <- tm_map(corpus_data, PlainTextDocument)
corpus_data <- tm_map(corpus_data, tolower)
corpus_data <- tm_map(corpus_data, textclean::replace_contraction)
corpus_data <- tm_map(corpus_data, function(x) {str_replace_all(x, "\\/", " ")})
corpus_data <- tm_map(corpus_data, removePunctuation)
corpus_data <- tm_map(corpus_data, stripWhitespace)
corpus_data <- tm_map(corpus_data, removeWords, stopwords("English"))
corpus_data <- tm_map(corpus_data, stemDocument)
dtm_data <- DocumentTermMatrix(corpus_data)
sparse_matrix <- removeSparseTerms(dtm_data, 0.995)
df_sparse <- as.data.frame(as.matrix(sparse_matrix))
colnames(df_sparse) <- make.names(colnames(df_sparse))
df_sparse$CustomerNo <- user_bought$CustomerNo


# feature selection -------------------------------------------------------

feature_data <- user_churn %>% 
  filter(!is.na(CustomerNo)) %>% 
  filter(NewUser == "no") %>%
  transmute(CustomerNo, Churn = if_else(Churn == "yes", 1, 0)) %>% 
  left_join(df_sparse, by = "CustomerNo") %>% 
  column_to_rownames(var = "CustomerNo")


# classification preparation ----------------------------------------------

set.seed(1999)
sample_split <- sample.split(feature_data$Churn, SplitRatio = .8)
train_data <- subset(feature_data, sample_split == TRUE)
test_data <- subset(feature_data, sample_split == FALSE)

train_data$Churn = as.factor(train_data$Churn)
test_data$Churn = as.factor(test_data$Churn)

tuning_model <- tuneRF(train_data[,-1], train_data[,1], ntreeTry = 500)
tuning_model


# classification result ---------------------------------------------------

RF_model <- randomForest(Churn ~ ., data = train_data, mtry = 37, ntree = 500)
predictRF <- predict(RF_model, newdata = test_data)
confusionMatrix(predictRF, reference = test_data$Churn)