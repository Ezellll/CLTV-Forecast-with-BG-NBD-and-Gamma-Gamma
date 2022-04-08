import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

############################################################################
# Görev 1: Veriyi Hazırlama
############################################################################


df_ = pd.read_csv("flo_data_20K.csv")
df = df_.copy()


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return round(low_limit), round(up_limit)


# Aykırı değerleri eşik değerleri ile değiştirme
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


replace_with_thresholds(df, "order_num_total_ever_online")
replace_with_thresholds(df, "order_num_total_ever_offline")
replace_with_thresholds(df, "customer_value_total_ever_offline")
replace_with_thresholds(df, "customer_value_total_ever_online")


df["Order_num_total_ever"] = df['order_num_total_ever_online'] + df['order_num_total_ever_offline']

df["Customer_value_total_ever"] = df['customer_value_total_ever_online'] + df['customer_value_total_ever_offline']

df.tail(10)

df.dtypes
for col in df.columns:
    if "date" in col:
        df[col] = pd.to_datetime(df[col])

########################################################################################
# Görev 2: CLTV Veri Yapısının Oluşturulması
#########################################################################################


cltv = pd.DataFrame()
df["last_order_date"].max()
analysis_date = dt.datetime(2021, 6, 1)


# cltv dataframe
cltv["customer_id"] = df["master_id"]

cltv["recency_cltv_weekly"] = ((df["last_order_date"] - df["first_order_date"]).dt.days) / 7

cltv["T_weekly"] = ((analysis_date - df["first_order_date"]).dt.days) / 7

cltv["frequency"] = df["Order_num_total_ever"]
cltv["monetary_cltv_avg"] = df["Customer_value_total_ever"]

#################################################################################
# Görev 3: BG/NBD, Gamma-Gamma Modellerinin Kurulması ve CLTV’nin Hesaplanması
#################################################################################

bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv['frequency'],
        cltv['recency_cltv_weekly'],
        cltv['T_weekly'])
# 3 ay içerisinde müşterilerden beklenen satın almaları tahmini

cltv["exp_sales_3_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(4 * 3,
                                                                                    cltv['frequency'],
                                                                                    cltv['recency_cltv_weekly'],
                                                                                    cltv['T_weekly'])
#  6 ay içerisinde müşterilerden beklenen satın almaları tahmini ve dataframe'e eklenmesi
# dataframe'ine ekleyiniz.
cltv["exp_sales_6_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(4 * 6,
                                                                                    cltv['frequency'],
                                                                                    cltv['recency_cltv_weekly'],
                                                                                    cltv['T_weekly'])

# 3. ay ve 6.aydaki en çok satın alım gerçekleştirecek 10 kişi

cltv.sort_values(by="exp_sales_3_month", ascending=False).head(10)
cltv.sort_values(by="exp_sales_6_month", ascending=False).head(10)

# Adım 2: Gamma-Gamma modelini fit edilmesi

ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv['frequency'], cltv['monetary_cltv_avg'])
# Average Profit
cltv["exp_average_value"] = ggf.conditional_expected_average_profit(cltv['frequency'],
                                                                    cltv['monetary_cltv_avg'])

# Adım 3: 6 aylık CLTV hesaplanması

cltv["cltv"] = ggf.customer_lifetime_value(bgf,
                                           cltv['frequency'],
                                           cltv['recency_cltv_weekly'],
                                           cltv['T_weekly'],
                                           cltv['monetary_cltv_avg'],
                                           time=6,  # 6 aylık
                                           freq="W",  # T'nin frekans bilgisi.
                                           discount_rate=0.01)

#cltv değerlerini standarlaştırıp scaled_cltv değişkeninin oluşturulması

scaler = MinMaxScaler(feature_range=(0,1))
cltv["scaled_cltv"] = scaler.fit_transform(cltv[["cltv"]])
cltv.sort_values(by= "cltv", ascending=False).head(20)

############################################################################
# Görev 4: CLTV Değerine Göre Segmentlerin Oluşturulması
############################################################################


# Adım 1: 6 aylık standartlaştırılmış CLTV'ye göre tüm müşterilerinizi 4 gruba
# (segmente) ayrılması ve grup isimlerini veri setine eklenmesi.

df["Segment"] = pd.qcut(cltv["cltv"], 4, labels=["D", "C", "B", "A"])
cltv["Segment"] = pd.qcut(cltv["cltv"], 4, labels=["D", "C", "B", "A"])
cltv.groupby("Segment").agg({"count", "mean", "sum"})


# CLTV skorlarına göre müşterileri 4 gruba ayırmak mantıklı mıdır?
# Daha az mı ya da daha çok mu olmalıdır? incelenmesi

# 2. durum
cltv["Segment"] = pd.qcut(cltv["cltv"], 2, labels=["B", "A"])
cltv.groupby("Segment").agg({"count", "mean", "sum"})

# 1. Durum
cltv["Segment"] = pd.qcut(cltv["cltv"], 6, labels=["F", "E", "D", "C", "B", "A"])
cltv.groupby("Segment").agg({"count", "mean", "sum"})

# Şirketin müşterileri sınıflandırma isteklerine göre değişebileceğini düşünüyorum.
# Bunu açıklamak için iki durum düşünelim. Birincisinde şirket gelecek satış politikası
# için kısıtlı imkanlara sahiptir ve yalnızca belirli ve az sayıda kullanıcıyla
# ilgilenebilecektir. Bu durumda grup sayısı artırılarak gruplarda bulunan insan sayısı
# azaltılmış olur ve böylelikle şirket stratejisini uygulamak için daha fazla imkana
# sahip olur. Diğer bir durumda ise şirketi daha az gruba ayırarak segmentler arasındaki fark artar
# ve şirket ayrılan iki grup için daha çok insan sayısına sahip ama daha az seçicilikle stratejisini
# belirlemelidir.


# Adım 3: 4 grup içerisinden seçeceğiniz 2 grup için yönetime kısa kısa 6
# aylık aksiyon önerilerinde bulununuz.


# A grubu:

# A grubunda  bulunan müşteriler 6 aylık satış beklentileri ve ortalama kazanç
# beklentileri diğer segmentlere göre yüksektir.
# Bu sebeple firma indirim tanımlama, hediye çeki ve insan kaynaklarını bu müşterilerine
# harcamayı tercih etmelidir. Böylelikle müşteriler kendilerini diğer gruplardaki müşterilerden
# üstün görür ve bu grupta kalmak için alışveriş alışkanlıklarını devam ettirirler.


# D Grubu:

# D grubunda bulunan müşteriler 6 aylık satış beklentisi ve ortalama kazanç beklentisi en düşük gruptur.
# Bu grubun haftalık recency T değerleri incelendiğinde diğer gruplara göre oldukça yüksektir.
# Bu durum müşterilerin uzun süredir firmadan alışveriş yapmayı tercih etmediğini gösterir.
# Bu durumda firma iki farklı strateji izleyebilir;
# ilk olarak, bu müşterilerin yeniden kazanılması için reklamlar ve mail aracılığı ile kampanya düzenlenerek
# firmadan tekrardan alışveriş alışkanlığını kazandırmak hedeflenebilir.
# Diğer bir durum firmada kısıtlı kaynak varsa bu gruba uyguluycağı kaynağı diğer gruplarda kullanarak daha verimli
# bir satış stratejisi izleyebilir.
