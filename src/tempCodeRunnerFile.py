### Câu 10: Tìm ra các seller (người bán) trong các phân khúc giá xe khác nhau (Cheap, Normal, Expensive, Very Expensive) dựa trên giá trị thị trường MMR
print("\n=== CÂU 10 ===")
query10= """
WITH car_value_group AS (
  SELECT *,
         CASE
           WHEN mmr < 5000 THEN 'Low'
           WHEN mmr BETWEEN 5000 AND 15000 THEN 'Mid'
           ELSE 'High'
         END AS value_segment
  FROM car_prices
  WHERE mmr IS NOT NULL AND sellingprice IS NOT NULL
        AND mmr > 0 AND sellingprice > 0
)
SELECT 
    seller,
    value_segment,
    COUNT(*) AS total_sales,
    ROUND(AVG(sellingprice), 2) AS avg_selling_price,
    ROUND(AVG(mmr), 2) AS avg_market_value,
    ROUND(AVG(sellingprice) / AVG(mmr), 2) AS price_ratio,
    CASE
        WHEN ROUND(AVG(sellingprice) / AVG(mmr), 2) < 0.75 THEN 'Very cheap'
        WHEN ROUND(AVG(sellingprice) / AVG(mmr), 2) >= 0.75 AND ROUND(AVG(sellingprice) / AVG(mmr), 2) < 0.95 THEN 'Cheap'
        WHEN ROUND(AVG(sellingprice) / AVG(mmr), 2) >= 0.95 AND ROUND(AVG(sellingprice) / AVG(mmr), 2) < 1.05 THEN 'Normal'
        WHEN ROUND(AVG(sellingprice) / AVG(mmr), 2) >= 1.05 AND ROUND(AVG(sellingprice) / AVG(mmr), 2) <= 1.25 THEN 'Expensive'
        WHEN ROUND(AVG(sellingprice) / AVG(mmr), 2) > 1.25 THEN 'Very expensive'
        ELSE 'Không xác định' 
    END AS seller_review
FROM car_value_group
GROUP BY seller, value_segment
HAVING COUNT(*) > 50

LIMIT 20
"""
spark.sql(query10).show()