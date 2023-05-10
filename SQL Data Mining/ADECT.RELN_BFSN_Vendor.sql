USE [ML]
GO
/****** Object:  StoredProcedure [ADECT].[S_Vendor_And_Bank]    Script Date: 10.05.2023 14:18:41 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
ALTER PROCEDURE [ADECT].[S_Vendor_And_Bank] AS BEGIN

TRUNCATE TABLE [ADECT].[RELN_BFSN_Vendor]

		--RELN Vendor Entries 
		INSERT INTO [ADECT].[RELN_BFSN_Vendor]
		(
		  Vendor_Number
		 ,Vendor_Name
		 ,Address
		 ,City
		 ,Country_Region_Code
		 ,Blocked
		 ,VAT_Registration_Number
		 ,Source_System
		 )
		SELECT DISTINCT
		  Vendor_Number
		 ,Vendor_Name
		 ,Address
		 ,City
		 ,Country_Region_Code
		 ,Blocked
		 ,VAT_Registration_Number
		 ,'RELN' AS Source_System
		FROM HUB.PAPO.RELN_Vendor
		

		-- BFSN Vendor Entries
		INSERT INTO [ADECT].[RELN_BFSN_Vendor]
		(
		  Vendor_Number
		 ,Vendor_Name
		 ,Address
		 ,City
		 ,Country_Region_Code
		 ,Blocked
		 ,VAT_Registration_Number
		 ,Source_System
		)
		SELECT DISTINCT
		  Vendor_Number
		 ,Vendor_Name
		 ,Address
		 ,City
		 ,Country_Region_Code
		 ,Blocked
		 ,VAT_Registration_Number
		 ,'BFSN' AS Source_System
		FROM [HUB].PAPO.BFSN_Outletcity_Metzingen_GmbH_Vendor bven
		

TRUNCATE TABLE [ADECT].[Vendor_Bank_Account]

		--RELN Vendor Bank Entries
		INSERT INTO [ADECT].[Vendor_Bank_Account]
		(
		Vendor_Number
		,IBAN
		,Bank_Account_Number
		,Bank_Branch_Number
		,Source_System
		)
		SELECT
		Vendor_Number
		,IBAN
		,Bank_Account_Number
		,Bank_Branch_Number
		,'RELN' AS Source_System
		FROM [HUB].[PAPO].[RELN_RE_Vendor_Bank_Account]
		

		--BFSN Vendor Bank Entries
		INSERT INTO [ADECT].[Vendor_Bank_Account]
		(
		Vendor_Number
		,IBAN
		,Bank_Account_Number
		,Bank_Branch_Number
		,Source_System
		)
		SELECT
		Vendor_Number
		,IBAN
		,Bank_Account_Number
		,Bank_Branch_Number
		,'RELN' AS Source_System
		FROM [HUB].[PAPO].[BFSN_Outletcity_Metzingen_GmbH_Vendor_Bank_Account]
		

TRUNCATE TABLE [ADECT].[MERGED_VENDOR_BANK]

INSERT INTO [ADECT].[MERGED_VENDOR_BANK]
	(
	Vendor_Number,
    Vendor_Name,
    Address,
    City,
    Country_Region_Code,
    Blocked,
    VAT_Registration_Number,
    Source_System,
    IBAN,
    Bank_Account_Number,
    Bank_Branch_Number,
	RowNumber
	)
	SELECT 
		Vendor_Number,
        Vendor_Name,
        Address,
        City,
        Country_Region_Code,
        Blocked,
        VAT_Registration_Number,
        Source_System,
        IBAN,
        Bank_Account_Number,
        Bank_Branch_Number,
        RowNumber
      FROM (
        SELECT 
			v.Vendor_Number AS Vendor_Number,
	        v.Vendor_Name AS Vendor_Name,
	        v.Address AS Address,
	        v.City AS City,
	        v.Country_Region_Code AS Country_Region_Code,
	        v.Blocked AS Blocked,
	        v.VAT_Registration_Number AS VAT_Registration_Number,
	        v.Source_System AS Source_System,
	        ISNULL(b.IBAN,'unknown') AS IBAN,
	        ISNULL(b.Bank_Account_Number, 'unknown') AS Bank_Account_Number,
	        ISNULL(b.Bank_Branch_Number, 'unknown') AS Bank_Branch_Number,
	        ROW_NUMBER() OVER (PARTITION BY v.Vendor_Number, b.IBAN ORDER BY b.Bank_Account_Number) AS RowNumber
        FROM [RELN_BFSN_Vendor] v
        LEFT JOIN (
            SELECT DISTINCT 
                Vendor_Number, 
                IBAN, 
                Bank_Account_Number, 
                Bank_Branch_Number 
            FROM [Vendor_Bank_Account] b
        ) b ON v.Vendor_Number = b.Vendor_Number
      ) AS T
	WHERE T.RowNumber = 1


--Fill empty cells in [Bank_Branch_Number] & [Bank_Account_Number] with data from [Vendor_Bank_Account] table
UPDATE [ADECT].[MERGED_VENDOR_BANK]
SET [Bank_Branch_Number] = vb.[Bank_Branch_Number],
    [Bank_Account_Number] = vb.[Bank_Account_Number]
FROM [ADECT].[MERGED_VENDOR_BANK] mvb
INNER JOIN [Vendor_Bank_Account] vb
ON mvb.[IBAN] = vb.[IBAN]
WHERE mvb.[Bank_Branch_Number] = '' OR mvb.[Bank_Account_Number] = ''

END