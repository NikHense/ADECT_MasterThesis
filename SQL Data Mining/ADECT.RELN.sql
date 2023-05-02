USE [ML]
GO
/****** Object:  StoredProcedure [ADECT].[S_RELN_PAYMENT]    Script Date: 02.05.2023 10:25:27 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO

ALTER PROCEDURE [ADECT].[S_RELN_PAYMENT] AS BEGIN

TRUNCATE TABLE [ADECT].[RELN_PAYMENT]

-- Payment Transaction, Payment Entry, Payment Line, Vendor & Vendor_ledger_entry
INSERT INTO [ADECT].[RELN_PAYMENT]
	(
	[Payment_Number]
      ,[Posting_Description_1]
      ,[Posting_Description_2]
      ,[Posting_Description_3]
      ,[Document_Number_external]
      ,[Document_Number_internal]
      --,[Contract_Number]
      ,[Gen_Jnl_Line_Number]
      ,[Line_Number]
      ,[ID_Vendor_Entry]
      ,[Object_Number]
      ,[Vendor_Number]
      ,[Name]
      ,[City]
      ,[Country_Region_Code]
      ,[Amount_Applied]
      ,[Amount_Initial]
      ,[Discount_Applied]
      ,[Discount_Allowed]
      ,[Discount_Rate]
      ,[Discount_Possible]
      --,[VAT_Rate]
      --,[VAT_Amount]
      ,[Payment_Method_Code]
      ,[Customer_Bank_Branch_Number]
      ,[Customer_Bank_Account_Number]
      ,[Customer_IBAN]
      ,[Vendor_Account_Holder]
      ,[Vendor_IBAN]
	  ,[Vendor_BIC]
      ,[Vendor_Bank_Account_Number]
      ,[Vendor_Bank_Branch_Number]
	  ,[Vendor_Bank_Origin]
      ,[Posting_Date]
      ,[Posting_Date_Changed]
      ,[Due_Date]
      ,[Last_Payment_Date]
      ,[Entry_Cancelled]
      ,[Jnl_Changed_By]
      ,[Jnl_Changed_On]
      --,[Blocked_Vendor]
      ,[Review_Status]
      ,[Created_By]
      ,[Vendor_Number_Name]
	  ,[Source_System]
      ,[Rownumber]
	  ,[Year]
      ,[Year-Month]
      ,[Mandant]
	)
SELECT
	pt.Case_Number AS Payment_Number,
	ISNULL(vle.Description_1, 'unknown') AS Posting_Description_1,
	ISNULL(vle.Description_2, 'unknown') AS Posting_Description_2,
	ISNULL(pl.Remittance_Inf, 'unknown') AS Posting_Description_3,
	
	ISNULL(pl.Remittance_Inf,'unknown') AS Document_Number_external, --need to extract number from text string
	ISNULL(pl.Document_Number,'unknown') AS Document_Number_internal,
	--ISNULL(pl.Contract_Number,'unknown') AS Contract_Number,

	ISNULL(pe.Serial_Number,0) AS Gen_Jnl_Line_Number,
	ISNULL(pl.Gen_Jnl_Line_Number, 0) AS Line_Number,
	ISNULL(pl.Serial_Number_Vendor_Entry,0) AS ID_Vendor_Entry, 
	ISNULL(pe.Object_Number,'unknown') AS Object_Number,

	ISNULL(pl.Vendor_Number_Customer_Number,'unknown') AS Vendor_Number,
	ISNULL(ven.Vendor_Name, 'unknown') AS Name,
	ISNULL(ven.City, 'unknown') AS City,
	ISNULL(ven.Country_Region_Code, 'unknown') AS Country_Region_Code,

	ISNULL(pl.Amount,0) AS Amount_Applied,
	ISNULL(pl.Original_Amount,0) AS Amount_Initial,
	ISNULL(pl.Discount,0) AS Discount_Applied,
	0 AS Discount_Allowed,
	0 AS Discount_Rate,
	0 AS Discount_Possible,
	--ISNULL(pl.VAT_Rate,0) AS VAT_Rate, --no need for VAT, since only empty cells
	--ISNULL(pl.VAT_Amount,0) AS VAT_Amount, --no need for VAT, since only empty cells
	ISNULL(pe.Payment_Method_Code, 'unkown') AS Payment_Method_Code,

	ISNULL(pe.Customer_Bank_Branch_Number,'unknown') AS Customer_Bank_Branch_Number,
	ISNULL(pe.Customer_Bank_Account_Number,'unknown') AS Customer_Bank_Account_Number,
	ISNULL(pe.Customer_IBAN,'unknown') AS Customer_IBAN,

	ISNULL(pl.Recipient_Account_Holder,'unknown') AS Vendor_Account_Holder,
	ISNULL(pl.Recipient_IBAN,'unknown') AS Vendor_IBAN,
	ISNULL(pl.Recipient_BIC_Code,'unknown') AS Vendor_BIC,
	ISNULL(pl.Recipient_Bank_Account_Number,'unknown') AS Vendor_Bank_Account_Number,
	ISNULL(pl.Recipient_Bank_Branch_Number,'unknown') AS Vendor_Bank_Branch_Number,
	ISNULL(LEFT(pl.Recipient_IBAN,2),'unknown') AS Vendor_Bank_Origin,

	pt.Posting_Date,
	pt.Posting_Date_Changed,
	pt.Execute_On AS Due_Date,
	ISNULL(pl.Last_Payment_Date, '1900-01-01') AS Last_Payment_Date,

	ISNULL(pt.Cancelled, 999) AS Entry_Cancelled,
	ISNULL(pl.JournLine_Manually_Changed_By, 'unknown') AS Jnl_Changed_By,
	ISNULL(pl.JournLine_Manually_Changed_On, '1900-01-01') AS Jnl_Changed_On,
	--ISNULL(ven.Blocked, 999) AS Blocked_Vendor,
	'Booked' AS Review_Status,
	pt.[User] AS Created_By,

	ISNULL(ven.Vendor_Number + ' ' + ven.Vendor_Name,'') AS Vendor_Number_Name,
	'RELN' AS Source_System,
	ROW_NUMBER() OVER (PARTITION BY pl.Vendor_Number_Customer_Number ORDER BY pl.Vendor_Number_Customer_Number) AS Rownumber,
	YEAR(pt.Posting_Date) AS [Year],
	CONVERT(NVARCHAR(20), ISNULL(pt.Posting_Date,'1900-01-01'), 126) AS [Year-Month],
	CASE	WHEN ISNULL(pe.Object_Number,'unknown') LIKE ('10%') THEN '01_Holy AG'	--Object_Number - start with 10...
			WHEN ISNULL(pe.Object_Number,'unknown') LIKE ('05%') THEN '02_Immobilien_Sonstige' 
			WHEN ISNULL(pe.Object_Number,'unknown') LIKE ('20%') THEN '02_Immobilien_Sonstige' 
			WHEN ISNULL(pe.Object_Number,'unknown') LIKE ('30%') THEN '02_Immobilien_Sonstige' 
			WHEN ISNULL(pe.Object_Number,'unknown') LIKE ('6%') THEN '02_Immobilien_Sonstige' 
			WHEN ISNULL(pe.Object_Number,'unknown') LIKE ('7%') THEN '02_Immobilien_Sonstige' 
			WHEN ISNULL(pe.Object_Number,'unknown') LIKE ('11%') THEN '03_OUTLETCITY_Online'
			WHEN ISNULL(pe.Object_Number,'unknown') LIKE ('13%') THEN '04_Gastro'
			WHEN ISNULL(pe.Object_Number,'unknown') LIKE ('01%') THEN '05_Holy_Verwaltungs_GmbH'
	ELSE 'unknown' END AS Mandant
	FROM [HUB].[PAPO].[RELN_RE_Payment_Transaction] pt --981
		LEFT JOIN [HUB].[PAPO].[RELN_RE_Payment_Entry] pe ON pt.Case_Number = pe.Payment_Transaction_Number 
		LEFT JOIN [HUB].[PAPO].[RELN_RE_Payment_Line] pl ON pl.Serial_Number_Payment_Entry = pe.Serial_Number AND pl.Payment_Transaction_Number = pt.Case_Number
		LEFT JOIN [ADECT].[RELN_BFSN_Vendor] ven ON ven.Source_System = 'RELN' AND ven.Vendor_Number = pl.Vendor_Number_Customer_Number
		LEFT JOIN [HUB].[PAPO].[RELN_Vendor_Ledger_Entry] vle ON pl.Serial_Number_Vendor_Entry = vle.[Entry Number] AND vle.Vendor_Number = pl.Vendor_Number_Customer_Number AND vle.Object_Number = pe.Object_Number AND vle.Document_Number = pl.Document_Number 
WHERE pl.Vendor_Number_Customer_Number NOT LIKE ('1%') AND pt.Posting_Date >= '2016-01-01' --Excluding all Vendor_Numbers that start with 1 (=Debitors) and invoices that are older than 2016

/*
UPDATE [ADECT].[RELN_PAYMENT]
SET Document_Number_external = LEFT(Document_Number_external, LEN (Document_Number_external) - 10) 

UPDATE [ADECT].[RELN_PAYMENT]
SET Document_Number_external = RIGHT(Document_Number_external, LEN (Document_Number_external) - 4)
*/
END

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
--Check how many observations are in the respective columns (after JOIN)
SELECT
	COUNT(*)
FROM [HUB].[PAPO].[RELN_RE_Payment_Entry] pe -- 5174
		LEFT JOIN [HUB].[PAPO].[RELN_RE_Payment_Transaction] pt ON pt.Case_Number = pe.Payment_Transaction_Number --5174

SELECT
	COUNT(*)
FROM [HUB].[PAPO].[RELN_RE_Payment_Line] pl --21528
	LEFT JOIN [HUB].[PAPO].[RELN_RE_Payment_Entry] pe ON pl.Serial_Number_Payment_Entry = pe.Serial_Number --21528
	LEFT JOIN [HUB].[PAPO].[RELN_RE_Payment_Transaction] pt ON pt.Case_Number = pe.Payment_Transaction_Number AND pl.Payment_Transaction_Number = pt.Case_Number --21528
	LEFT JOIN [ADECT].[RELN_BFSN_Vendor] ven ON ven.Source_System = 'RELN' AND ven.Vendor_Number = pl.Vendor_Number_Customer_Number --21528
	LEFT JOIN [HUB].[PAPO].[RELN_Vendor_Ledger_Entry] vle ON vle.Document_Number = pl.Document_Number AND vle.Vendor_Number = ven.Vendor_Number AND vle.Object_Number = pe.Object_Number AND pl.Serial_Number_Vendor_Entry = vle.[Entry Number] --21528
WHERE pl.Vendor_Number_Customer_Number NOT LIKE ('1%') --Excluding all Vendor_Numbers that start with 1 (=Debitors)	
		
SELECT
COUNT(*)
FROM [HUB].[PAPO].[RELN_RE_Payment_Transaction] pt -- 1086
		LEFT JOIN [HUB].[PAPO].[RELN_RE_Payment_Entry] pe ON pe.Payment_Transaction_Number = pt.Case_Number --5243


-- Check number of obs. that are in Entry but not in Transaction
SELECT *
FROM [HUB].[PAPO].[RELN_RE_Payment_Entry]
WHERE Payment_Transaction_Number NOT IN (SELECT Case_Number FROM [HUB].[PAPO].[RELN_RE_Payment_Transaction] ) --2

--Check number of obs. that are in Transaction but not in Entry
--THere is not Entry for those 71 obs. since they are cancelled (Cancelled = 1)
SELECT *
FROM [HUB].[PAPO].[RELN_RE_Payment_Transaction]
WHERE Case_Number NOT IN (SELECT Payment_Transaction_Number FROM [HUB].[PAPO].[RELN_RE_Payment_Entry]) --71

----------------------------------------------------------------------------------------------------------------
--Check how many entries there are in vle to merge
SELECT *
FROM [HUB].[PAPO].[RELN_Vendor_Ledger_Entry]
WHERE Document_Number IN (SELECT Document_Number FROM [HUB].[PAPO].[RELN_RE_Payment_Line]) --2.725

SELECT *
FROM [HUB].[PAPO].[RELN_Vendor_Ledger_Entry]
WHERE [Entry Number] IN (SELECT Serial_Number_Vendor_Entry FROM [HUB].[PAPO].[RELN_RE_Payment_Line]) --2.706

SELECT *
FROM [HUB].[PAPO].[RELN_Vendor_Ledger_Entry]
WHERE Vendor_Number IN (SELECT Vendor_Number_Customer_Number FROM [HUB].[PAPO].[RELN_RE_Payment_Line]) --35.075
----------------------------------------------------------------------------------------------------------------
--Check whether suppliers are inluded (should not be the case)
SELECT
COUNT (*)
FROM [HUB].[PAPO].[RELN_RE_Payment_Transaction] pt --981
		LEFT JOIN [HUB].[PAPO].[RELN_RE_Payment_Entry] pe ON pt.Case_Number = pe.Payment_Transaction_Number 
		LEFT JOIN [HUB].[PAPO].[RELN_RE_Payment_Line] pl ON pl.Serial_Number_Payment_Entry = pe.Serial_Number AND pl.Payment_Transaction_Number = pt.Case_Number
		LEFT JOIN [ADECT].[RELN_BFSN_Vendor] ven ON ven.Source_System = 'RELN' AND ven.Vendor_Number = pl.Vendor_Number_Customer_Number
		LEFT JOIN [HUB].[PAPO].[RELN_Vendor_Ledger_Entry] vle ON pl.Serial_Number_Vendor_Entry = vle.[Entry Number] AND vle.Vendor_Number = pl.Vendor_Number_Customer_Number AND vle.Object_Number = pe.Object_Number AND vle.Document_Number = pl.Document_Number
WHERE pl.Vendor_Number_Customer_Number NOT LIKE ('1%') --12.970
--WHERE pl.Vendor_Number_Customer_Number  LIKE ('1%') --7.176
--Difference of the two WHERE-clauses are the observations (68, see also SELECT above) with Serial Number = 0