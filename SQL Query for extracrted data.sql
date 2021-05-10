SELECT DISTINCT(yb.business_id),
	   name,
	   address,
	   city,
	   state,
	   postal_code,
	   yb.stars,
	   review_count,
	   review_id,	
	   user_id,
	   yr.stars 'user_given_stars',
	   text,
	   useful,
	   funny,
	   cool,
	   CASE  
			WHEN yr.stars <= 3 THEN 'Negative'
			WHEN yr.stars > 3 THEN 'Positive'
		END Polarity
FROM dbo.yelp_business yb INNER JOIN  dbo.yelp_review yr
	ON yb.business_id = yr.business_id
WHERE state = 'PA' 
	and text IS NOT NULL
	and is_open = 1
	and postal_code IS NOT NULL
	and city LIKE '%Pittsburgh%'
	and yb.categories LIKE '%restaurant%'
ORDER BY postal_code