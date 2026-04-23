# Errors
These are examples where i disagree with the gold answer.
---
Microbreweries should be aggregated by region.

6. Identify the top 30 microbreweries with the smallest beer catalogs.
     iter=8  dur=13.1s  tools: search_rules → list_schemas → list_tables → describe_table → 
describe_table → run_query → run_query → submit_answer
     error: Results do not match
     submitted: SELECT b.name, b.city, b.state, COUNT(be.id) AS beer_count
FROM CraftBeer.breweries b
LEFT JOIN CraftBeer.beers be ON b.id = be.brewery_id
WHERE b.city IS NOT NULL AND b.state IS NOT NULL
GROUP BY b.name, b.city, b.state
HAVING COUNT(be.id) < 3
ORDER BY beer_count ASC, b.name ASC
LIMIT 30;
     gold:      SELECT br.id, br.name, COUNT(b.id) as beer_count FROM CraftBeer.breweries br JOIN 
CraftBeer.beers b ON br.id = b.brewery_id GROUP BY br.id, br.name HAVING COUNT(b.id) < 3 ORDER BY 
beer_count ASC, br.name LIMIT 30
     llm_category: wrong_groupby
     llm_reason: The submitted query groups by brewery name, city, and state instead of brewery id 
and name, causing different aggregation and results.