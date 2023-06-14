USE [ML]
GO
/****** Object:  StoredProcedure [ADECT].[S_BFSN_PAYMENT]    Script Date: 02.05.2023 10:33:39 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO

ALTER PROCEDURE [ADECT].[S_BFSN_PAYMENT] AS BEGIN

TRUNCATE TABLE [ADECT].[BFSN_PAYMENT]

-- Payment Proposal, Payment Proposal Head, Payment Proposal Line, Vendor
INSERT INTO [ADECT].[BFSN_PAYMENT]
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
	pp.Journal_Batch_Name AS Payment_Number,
	pl.Payment_Text AS Posting_Description_1,
	pl.Payment_Description AS Posting_Description_2,
	'' AS Posting_Description_3,
	ISNULL(pl.External_Document_Number, 'unknown') AS Document_Number_external,
	ISNULL(pl.Applies_To_Doc_Number, 'unknown') AS Document_Number_internal,
	--'' AS Contract_Number,
	ph.Gen_Journal_Line AS Gen_Jnl_Line_Number,
	pl.Line_Number AS Line_Number,
	pl.ID_Applied_Entry AS ID_Vendor_Entry,
	'11005' AS Object_Number,
	ph.Account_Number AS Vendor_Number,
	ph.Name AS Name,
	ph.City AS City,
	ph.Country_Region_Code AS Country_Region_Code,
	pl.Posting_Applied_Amount AS Amount_Applied,
	pl.Original_Amount AS Amount_Initial,
	pl.Posting_Payment_Discount AS Discount_Applied,
	pl.Payment_Discount_Allowed AS Discount_Allowed,
	pl.Original_Pmt_Discount AS Discount_Rate,
	pl.Payment_Discount_Possible AS Discount_Possible,
	--0 AS VAT_Rate,
	--0 AS VAT_Amount,
	ph.Orig_Payment_Method AS Payment_Method_Code,
	ph.Orderer_Bank_Branch_Number AS Customer_Bank_Branch_Number,
	ph.Orderer_Bank_Account_Number AS Customer_Bank_Account_Number,
	ph.Orderer_Bank_IBAN_Code AS Customer_IBAN,
	ph.Bank_Account_Owner AS Vendor_Account_Holder,
	ph.Bank_IBAN_Code AS Vendor_IBAN,
	ph.Bank_BIC_Code AS Vendor_BIC,
	ph.Bank_Account_Number AS Vendor_Bank_Account_Number,
	ph.Bank_Branch_Number AS Vendor_Bank_Branch_Number,
	LEFT(ph.Bank_IBAN_Code,2) AS Vendor_Bank_Origin,
	pp.Posting_Date AS Posting_Date,
	0 AS Posting_Date_Changed,
	pl.Due_Date AS Due_Date,
	'1900-01-01' AS Last_Payment_Date,
	CASE	WHEN pp.Description LIKE ('Gelöschter Zahlungslauf') THEN 1
			WHEN pp.Description NOT LIKE ('Gelöschter Zahlungslauf') THEN 0
			ELSE 'unknown' END AS Entry_Cancelled,
	'' AS Jnl_Changed_By,
	'1900-01-01' AS Jnl_Changed_On,
	--ISNULL(ven.Blocked,999) AS Blocked_Vendor,
	CASE	WHEN pp.Posting_Date >= DATEADD(day, -7, GETDATE())  THEN 'Proposal'
			ELSE 'Booked' END AS Review_Status,
	pp.User_ID AS Created_By,
	ISNULL(ven.Vendor_Number + ' ' + ven.Vendor_Name, '') AS Vendor_Number_Name,
	'BFSN' AS Source_System,
	ROW_NUMBER() OVER (PARTITION BY ph.Account_Number ORDER BY ph.Account_Number) AS Rownumber,
	YEAR(ISNULL(ph.Posting_Date,'1900-01-01')) AS [Year],
	CONVERT(NVARCHAR(7), ISNULL(ph.Posting_Date,'1900-01-01'), 126) AS [Year-Month],
	'03_OUTLETCITY_Online' AS Mandant

FROM [HUB].[PAPO].[BFSN_Outletcity_Metzingen_GmbH_Payment_Proposal] pp
		LEFT JOIN [HUB].[PAPO].[BFSN_Outletcity_Metzingen_GmbH_Payment_Proposal_Head] ph ON pp.Journal_Batch_Name = ph.Gen_Journal_Batch
		LEFT JOIN [HUB].[PAPO].[BFSN_Outletcity_Metzingen_GmbH_Payment_Proposal_Line] pl ON ph.Gen_Journal_Batch = pl.Journal_Batch_Name AND ph.Gen_Journal_Line = pl.Journal_Line_Number
		LEFT JOIN [ADECT].[RELN_BFSN_Vendor] ven ON ven.Source_System = 'BFSN' AND ven.Vendor_Number = pl.Source_Number
WHERE pl.Account_Type = 2 AND pp.Posting_Date >= '2016-01-01'

END
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
--Check how many observations are in the respective columns (after JOIN)
SELECT
	COUNT(*)
FROM [HUB].[PAPO].[BFSN_Outletcity_Metzingen_GmbH_Payment_Proposal_Head] ph
	 LEFT JOIN [HUB].[PAPO].[BFSN_Outletcity_Metzingen_GmbH_Payment_Proposal] pp ON pp.Journal_Batch_Name = ph.Gen_Journal_Batch
	 

SELECT
	COUNT(*)
FROM [HUB].[PAPO].[BFSN_Outletcity_Metzingen_GmbH_Payment_Proposal_Line] pl
	LEFT JOIN [HUB].[PAPO].[BFSN_Outletcity_Metzingen_GmbH_Payment_Proposal_Head] ph ON ph.Gen_Journal_Batch = pl.Journal_Batch_Name AND ph.Gen_Journal_Line = pl.Journal_Line_Number
	LEFT JOIN [HUB].[PAPO].[BFSN_Outletcity_Metzingen_GmbH_Payment_Proposal] pp ON pp.Journal_Batch_Name = ph.Gen_Journal_Batch
	LEFT JOIN [ADECT].[RELN_BFSN_Vendor] ven ON ven.Source_System = 'BFSN' AND ven.Vendor_Number = pl.Account_Number
WHERE pl.Account_Type = 2 
