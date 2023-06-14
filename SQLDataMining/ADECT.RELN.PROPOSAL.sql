USE [ML]
GO
/****** Object:  StoredProcedure [ADECT].[S_RELN_PAYMENT_PROPOSAL]    Script Date: 04.05.2023 11:29:50 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO


ALTER PROCEDURE [ADECT].[S_RELN_PAYMENT_PROPOSAL] AS BEGIN

TRUNCATE TABLE [ADECT].[RELN_PAYMENT_PROPOSAL]

-- Payment Proposal Entry, Payment Proposal Line, Payment Proposal 
INSERT INTO [ADECT].[RELN_PAYMENT_PROPOSAL]
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
	pp.Payment_Proposal_Number AS Payment_Number,
	ISNULL(vle.Description_1, 'unknown') AS Posting_Description_1,
	ISNULL(vle.Description_2, 'unknown') AS Posting_Description_2,
	ISNULL(ppl.Remittance_Inf, 'unknown') AS Posting_Description_3,

	ISNULL(ppl.Remittance_Inf,'unknown') AS Document_Number_external, --need to extract number from text string
	ISNULL(ppl.Document_Number,'unknown') AS Document_Number_internal,
	--ISNULL(pl.Contract_Number,'unknown') AS Contract_Number,

	ISNULL(ppe.Serial_Number,0) AS Gen_Jnl_Line_Number,
	ISNULL(ppl.Gen_Jnl_Line_Number, 0) AS Line_Number,
	ISNULL(ppl.Serial_Number_Debt_CredEntry,0) AS ID_Vendor_Entry, 
	ISNULL(ppe.Object_Number,'unknown') AS Object_Number,

	ISNULL(ppl.Number_Customer_Vendor,'unknown') AS Vendor_Number,
	ISNULL(ven.Vendor_Name, 'unknown') AS Name,
	ISNULL(ven.City, 'unknown') AS City,
	ISNULL(ven.Country_Region_Code, 'unknown') AS Country_Region_Code,

	ISNULL(ppl.Amount,0) AS Amount_Applied,
	ISNULL(ppl.Original_Amount,0) AS Amount_Initial,
	ISNULL(ppl.Discount,0) AS Discount_Applied,
	0 AS Discount_Allowed,
	0 AS Discount_Rate,
	0 AS Discount_Possible,
	--ISNULL(pl.VAT_Rate,0) AS VAT_Rate,
	--ISNULL(pl.VAT_Amount,0) AS VAT_Amount,
	ISNULL(ppe.Payment_Method_Code, 'unkown') AS Payment_Method_Code,

	ISNULL(ppe.Customer_Bank_Branch_Number,'unknown') AS Customer_Bank_Branch_Number,
	ISNULL(ppe.Customer_Bank_Account_Number,'unknown') AS Customer_Bank_Account_Number,
	ISNULL(ppe.Customer_IBAN,'unknown') AS Customer_IBAN,

	ISNULL(ppl.Recipient_Account_Holder,'unknown') AS Vendor_Account_Holder,
	ISNULL(ppl.Recipient_IBAN,'unknown') AS Vendor_IBAN,
	ISNULL(ppl.Recipient_BIC_Code,'unknown') AS Vendor_BIC,
	ISNULL(ppl.Recipient_Bank_Account_Number,'unknown') AS Vendor_Bank_Account_Number,
	ISNULL(ppl.Recipient_Bank_Branch_Number,'unknown') AS Vendor_Bank_Branch_Number,
	ISNULL(LEFT(ppl.Recipient_IBAN,2),'unknown') AS Vendor_Bank_Origin,

	pp.Posting_Date,
	0 AS Posting_Date_Changed,
	pp.Last_Due_Date AS Due_Date,
	ISNULL(ppl.Last_Payment_Date, '1900-01-01') AS Last_Payment_Date,

	999 AS Entry_Cancelled,
	ISNULL(ppl.JournLine_Manually_Changed_By, 'unknown') AS Jnl_Changed_By,
	ISNULL(ppl.JournLine_Manually_Changed_On, '1900-01-01') AS Jnl_Changed_On,
	--ISNULL(ven.Blocked, 999) AS Blocked_Vendor,
	'Proposal' AS Review_Status,
	pp.Created_By AS Created_By,

	ISNULL(ven.Vendor_Number + ' ' + ven.Vendor_Name,'') AS Vendor_Number_Name,
	'RELN' AS Source_System,
	ROW_NUMBER() OVER (PARTITION BY ppl.Number_Customer_Vendor ORDER BY ppl.Number_Customer_Vendor) AS Rownumber,
	YEAR(pp.Posting_Date) AS [Year],
	CONVERT(NVARCHAR(20), ISNULL(pp.Posting_Date,'1900-01-01'), 126) AS [Year-Month],
	CASE	WHEN ISNULL(ppe.Object_Number,'unknown') LIKE ('10%') THEN '01_Holy AG'	--Object_Number - start with 10...
			WHEN ISNULL(ppe.Object_Number,'unknown') LIKE ('05%') THEN '02_Immobilien_Sonstige' 
			WHEN ISNULL(ppe.Object_Number,'unknown') LIKE ('20%') THEN '02_Immobilien_Sonstige' 
			WHEN ISNULL(ppe.Object_Number,'unknown') LIKE ('30%') THEN '02_Immobilien_Sonstige' 
			WHEN ISNULL(ppe.Object_Number,'unknown') LIKE ('6%') THEN '02_Immobilien_Sonstige' 
			WHEN ISNULL(ppe.Object_Number,'unknown') LIKE ('7%') THEN '02_Immobilien_Sonstige' 
			WHEN ISNULL(ppe.Object_Number,'unknown') LIKE ('11%') THEN '03_OUTLETCITY_Online'
			WHEN ISNULL(ppe.Object_Number,'unknown') LIKE ('13%') THEN '04_Gastro'
			WHEN ISNULL(ppe.Object_Number,'unknown') LIKE ('01%') THEN '05_Holy_Verwaltungs_GmbH'
	ELSE 'unknown' END AS Mandant

FROM [HUB].[PAPO].[RELN_RE_Payment_Proposal] pp
		LEFT JOIN [HUB].[PAPO].[RELN_RE_Payment_Proposal_Entry] ppe ON pp.Payment_Proposal_Number = ppe.Number_Payment_Proposal 
		LEFT JOIN [HUB].[PAPO].[RELN_RE_Payment_Proposal_Line] ppl ON ppl.Serial_Number_Paym_Proposal_Entry = ppe.Serial_Number AND ppl.Number_Payment_Proposal = pp.Payment_Proposal_Number
		LEFT JOIN [ADECT].[RELN_BFSN_Vendor] ven ON ven.Source_System = 'RELN' AND ven.Vendor_Number = ppl.Number_Customer_Vendor
		LEFT JOIN [HUB].[PAPO].[RELN_Vendor_Ledger_Entry] vle ON ppl.Serial_Number_Debt_CredEntry = vle.[Entry Number] AND vle.Vendor_Number = ppl.Number_Customer_Vendor AND vle.Object_Number = ppe.Object_Number AND vle.Document_Number = ppl.Document_Number 
WHERE ppl.Number_Customer_Vendor NOT LIKE ('1%') AND pp.Posting_Date >= DATEADD(DAY, -7, GETDATE()) -- just get the last 7 (rolling) days

END
