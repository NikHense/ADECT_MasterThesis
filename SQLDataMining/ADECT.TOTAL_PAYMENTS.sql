USE [ML]
GO
/****** Object:  StoredProcedure [ADECT].[S_TOTAL_PAYMENTS]    Script Date: 04.05.2023 11:30:17 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO


ALTER PROCEDURE [ADECT].[S_TOTAL_PAYMENTS] AS BEGIN

TRUNCATE TABLE [ADECT].[TOTAL_PAYMENTS] 

--RELN_PAYMENT, BFSN_PAYMENT
INSERT INTO [ADECT].[TOTAL_PAYMENTS]
(
	[Payment_Number]
	,[Posting_Description_1]
    ,[Posting_Description_2]
    ,[Posting_Description_3]
    ,[Document_Number_external]
    ,[Document_Number_internal]
    --,[Contract_Number] --> exclude from consideration, since there is no contract number for any (accounts payable) vendor
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
	reln.Payment_Number 
    ,reln.Posting_Description_1
    ,reln.Posting_Description_2
    ,reln.Posting_Description_3
    ,reln.Document_Number_external
    ,reln.Document_Number_internal
    --,reln.Contract_Number --> exclude from consideration, since there is no contract number for any (accounts payable) vendor
    ,reln.Gen_Jnl_Line_Number
    ,reln.Line_Number
    ,reln.ID_Vendor_Entry
    ,reln.Object_Number
    ,reln.Vendor_Number
    ,reln.[Name]
    ,reln.City
    ,reln.Country_Region_Code
    ,reln.Amount_Applied
    ,reln.Amount_Initial
    ,reln.Discount_Applied
    ,reln.Discount_Allowed
    ,reln.Discount_Rate
    ,reln.Discount_Possible
    --,reln.VAT_Rate
    --,reln.VAT_Amount
    ,reln.Payment_Method_Code
    ,reln.Customer_Bank_Branch_Number
    ,reln.Customer_Bank_Account_Number
    ,reln.Customer_IBAN
    ,reln.Vendor_Account_Holder
    ,reln.Vendor_IBAN
	,reln.Vendor_BIC
    ,reln.Vendor_Bank_Account_Number
    ,reln.Vendor_Bank_Branch_Number
	,reln.Vendor_Bank_Origin
    ,reln.Posting_Date
    ,reln.Posting_Date_Changed
    ,reln.Due_Date
    ,reln.Last_Payment_Date
    ,reln.Entry_Cancelled
    ,reln.Jnl_Changed_By
    ,reln.Jnl_Changed_On
    --,reln.Blocked_Vendor
    ,reln.Review_Status
    ,reln.Created_By
    ,reln.Vendor_Number_Name
	,reln.Source_System
    ,reln.Rownumber
	,reln.[Year]
    ,reln.[Year-Month]
    ,reln.Mandant

FROM [ADECT].[RELN_PAYMENT] reln
UNION 
SELECT 
	bfsn.Payment_Number 
    ,bfsn.Posting_Description_1
    ,bfsn.Posting_Description_2
    ,bfsn.Posting_Description_3
    ,bfsn.Document_Number_external
    ,bfsn.Document_Number_internal
    --,bfsn.Contract_Number --> exclude from consideration, since there is no contract number for any (accounts payable) vendor
    ,bfsn.Gen_Jnl_Line_Number
    ,bfsn.Line_Number
    ,bfsn.ID_Vendor_Entry
    ,bfsn.Object_Number
    ,bfsn.Vendor_Number
    ,bfsn.[Name]
    ,bfsn.City
    ,bfsn.Country_Region_Code
    ,bfsn.Amount_Applied
    ,bfsn.Amount_Initial
    ,bfsn.Discount_Applied
    ,bfsn.Discount_Allowed
    ,bfsn.Discount_Rate
    ,bfsn.Discount_Possible
    --,bfsn.VAT_Rate
    --,bfsn.VAT_Amount
    ,bfsn.Payment_Method_Code
    ,bfsn.Customer_Bank_Branch_Number
    ,bfsn.Customer_Bank_Account_Number
    ,bfsn.Customer_IBAN
    ,bfsn.Vendor_Account_Holder
    ,bfsn.Vendor_IBAN
	,bfsn.Vendor_BIC
    ,bfsn.Vendor_Bank_Account_Number
    ,bfsn.Vendor_Bank_Branch_Number
	,bfsn.Vendor_Bank_Origin
    ,bfsn.Posting_Date
    ,bfsn.Posting_Date_Changed
    ,bfsn.Due_Date
    ,bfsn.Last_Payment_Date
    ,bfsn.Entry_Cancelled
    ,bfsn.Jnl_Changed_By
    ,bfsn.Jnl_Changed_On
    --,bfsn.Blocked_Vendor
    ,bfsn.Review_Status
    ,bfsn.Created_By
    ,bfsn.Vendor_Number_Name
	,bfsn.Source_System
    ,bfsn.Rownumber
	,bfsn.[Year]
    ,bfsn.[Year-Month]
    ,bfsn.Mandant

FROM [ADECT].[BFSN_PAYMENT] bfsn
UNION 
SELECT 
	pp.Payment_Number 
    ,pp.Posting_Description_1
    ,pp.Posting_Description_2
    ,pp.Posting_Description_3
    ,pp.Document_Number_external
    ,pp.Document_Number_internal
    --,pp.Contract_Number --> exclude from consideration, since there is no contract number for any (accounts payable) vendor
    ,pp.Gen_Jnl_Line_Number
    ,pp.Line_Number
    ,pp.ID_Vendor_Entry
    ,pp.Object_Number
    ,pp.Vendor_Number
    ,pp.[Name]
    ,pp.City
    ,pp.Country_Region_Code
    ,pp.Amount_Applied
    ,pp.Amount_Initial
    ,pp.Discount_Applied
    ,pp.Discount_Allowed
    ,pp.Discount_Rate
    ,pp.Discount_Possible
    --,pp.VAT_Rate
    --,pp.VAT_Amount
    ,pp.Payment_Method_Code
    ,pp.Customer_Bank_Branch_Number
    ,pp.Customer_Bank_Account_Number
    ,pp.Customer_IBAN
    ,pp.Vendor_Account_Holder
    ,pp.Vendor_IBAN
	,pp.Vendor_BIC
    ,pp.Vendor_Bank_Account_Number
    ,pp.Vendor_Bank_Branch_Number
	,pp.Vendor_Bank_Origin
    ,pp.Posting_Date
    ,pp.Posting_Date_Changed
    ,pp.Due_Date
    ,pp.Last_Payment_Date
    ,pp.Entry_Cancelled
    ,pp.Jnl_Changed_By
    ,pp.Jnl_Changed_On
    --,pp.Blocked_Vendor
    ,pp.Review_Status
    ,pp.Created_By
    ,pp.Vendor_Number_Name
	,pp.Source_System
    ,pp.Rownumber
	,pp.[Year]
    ,pp.[Year-Month]
    ,pp.Mandant

FROM [ADECT].[RELN_PAYMENT_PROPOSAL] pp

--------------------------------------------------------------------------------------------------------
--Update Country_Region_Code spelling
UPDATE [ADECT].[TOTAL_PAYMENTS]
SET Country_Region_Code = 
	CASE Country_Region_Code 
	WHEN 'I' THEN 'IT'
	WHEN 'F' THEN 'FR'
	WHEN 'B' THEN 'BE'
	WHEN 'EE' THEN 'EST'
	WHEN 'BG' THEN 'BGR'
	WHEN 'PA' THEN 'PAN'
	WHEN 'FI' THEN 'FIN'
	WHEN 'LU' THEN 'LUX'
	WHEN 'UK' THEN 'GB'
	WHEN 'SI' THEN 'SVN'
	WHEN 'AE' THEN 'UAE'
	WHEN 'SG' THEN 'SGP'
	WHEN 'SE' THEN 'SWE'
	WHEN 'IE' THEN 'IRL'
	WHEN 'CA' THEN 'CAN'
	ELSE Country_Region_Code END
WHERE Country_Region_Code IN ('I', 'F', 'B', 'EE', 'BG', 'PA', 'FI', 'LU', 'UK', 'SI', 'AE', 'SG', 'SE', 'IE', 'CA')

-- Update Country_Region_Code based on City-Name
UPDATE [ADECT].[TOTAL_PAYMENTS]
SET Country_Region_Code = 
	CASE 
		WHEN City IN ('London', 'United Kingdom') THEN 'GB'
		WHEN City IN ('Beernem') THEN 'BE'
		WHEN City IN ('Dublin') THEN 'IRL'
		WHEN City IN ('Montreal, Quebec') THEN 'CAN'
		WHEN City IN ('Metzingen', 
						'Riedlingen',
						'Landshut',
						'Berlin',
						'Pfullingen',
						'Reutlingen',
						'Fellbach',
						'Philippsthal',
						'Hausen am Bach',
						'München',
						'Freiburg',
						'Bielefeld',
						'Emmering',
						'Schorndorf',
						'Künzing',
						'Martinsried',
						'Stuttgart',
						'Konstanz',
						'Bösenreutin',
						'Viersen',
						'Karlsbad',
						'Seevetal',
						'Taufkirchen',
						'Filderstadt',
						'Oberviechtach',
						'Pliezhausen') THEN 'DE'
		WHEN City IN ('Wien') THEN 'AT'
		WHEN City IN ('Alicante') THEN 'ES'
		WHEN City IN ('Rigas rajons') THEN 'LVA'
		ELSE Country_Region_Code
END

--Change spelling of specific City
UPDATE [ADECT].[TOTAL_PAYMENTS]
SET City = 
CASE 
	WHEN City IN ('Düsseldorg') THEN 'Düsseldorf'
	WHEN City IN ('Frankfurt', 'Frankfurt/Main') THEN 'Frankfurt am Main'
	WHEN City IN ('Copenhagen', 'Copenhagen K') THEN 'Kopenhagen'
	WHEN City IN ('Holzwickede', 'Dortmund-Holzwickede') THEN 'Dortmund'
	WHEN City IN ('München ', 'München-Flughafen') THEN 'München'
	WHEN City IN ('Erkrath ') THEN 'Erkrath'
	WHEN City IN ('Stuttgart-Degerloch', 'Stuttgart-Flughafen') THEN 'Stuttgart'
	WHEN City IN ('Kirchheim', 'Kirchheim unter Teck') THEN 'Kirchheim/Teck'
	WHEN City IN ('Walddorfhässlach') THEN 'Walddorfhäslach'
	WHEN City IN ('Mainz-Kastel') THEN 'Mainz'
	WHEN City IN ('Dubai, United Arab Emirates') THEN 'Dubai'
	WHEN City IN ('Aarhus N', 'ARHUS', 'Aarhus c') THEN 'Aarhus'
	WHEN City IN ('Ireland', 'Dublin D02X525, Ireland', 'Dublin 18 Irland') THEN 'Dublin'
	WHEN City IN ('SR Leiden ') THEN 'SR Leiden'
	WHEN City IN ('Taufkirchen/Pram') THEN 'Taufkirchen'
	WHEN City IN ('Paris - France', 'Paris – France', 'Paris 2') THEN 'Paris'
	WHEN City IN ('CK Amsterdam', 'GW Amsterdam', 'MS Halfweg (Amsterdam)', 'DJ Amsterdam') THEN 'Amsterdam'
	WHEN City IN ('Schwarzach') THEN 'Schwarzach am Main'
	WHEN City IN ('Feldkirchen-Westerham') THEN 'Feldkirchen'
	WHEN City IN ('Kirn / Nahe') THEN 'Kirn'
	WHEN City IN ('San Mauro Pascoli (FC)') THEN 'San Mauro Pascoli'
	WHEN City IN ('Planegg/Martinsried', 'Martinsried') THEN 'Planegg'
	WHEN City IN ('Garmisch-Partenkirchenu') THEN 'Garmisch-Partenkirchen'
	WHEN City IN ('Eningen', 'Eningen u. A.') THEN 'Eningen unter Achalm'
	WHEN City IN ('Rosenheim ') THEN 'Rosenheim'
	WHEN City IN ('Neuenstadt') THEN 'Neuenstadt am Kocher'
	WHEN City IN (' Empfingen') THEN 'Empfingen'
	WHEN City IN ('Leinfelder-Echterdingen') THEN 'Leinfelden-Echterdingen'
	WHEN City IN ('Helsinki, Finnland') THEN 'Helsinki'
	WHEN City IN ('Baden Baden') THEN 'Baden-Baden'
	WHEN City IN ('DD Weert') THEN 'Weert'
	WHEN City IN ('Kolding, Danmark') THEN 'Kolding'
	WHEN City IN ('Kusterdingen/Wankheim') THEN 'Kusterdingen-Wankheim'
	WHEN City IN ('Ditzingen-Schöckingen') THEN 'Ditzingen'
	WHEN City IN ('Herning, Denmark') THEN 'Herning'
	WHEN City IN ('Capalle, Firenze', 'Firenze', 'Scandicci Firenze') THEN 'Florenz'
	WHEN City IN ('HA Waalwijk', 'KG Waalwijk') THEN 'Waalwijk'
	WHEN City IN ('Milano (MI)', 'Milano', 'Milan') THEN 'Mailand'
	WHEN City IN ('HC Rotterdam') THEN 'Rotterdam'
	WHEN City IN ('Fidenze (Parma) Italy') THEN 'Fidenza'
	WHEN City IN ('Herford ') THEN 'Herford'
	WHEN City IN ('HC Rotterdam') THEN 'Rotterdam'
	WHEN City IN ('Fidenze (Parma) Italy') THEN 'Fidenza'
	WHEN City IN ('London SE1 3TY', 'London W1W 7PA', 'The Copperworks, London', 'Camden, London', 'London United Kingdom' ) THEN 'London'
	WHEN City IN ('Krefeld ') THEN 'Krefeld'
	WHEN City IN ('Martorell _Barcelona') THEN 'Barcelona'
	WHEN City IN ('Pforzheim-Mäuerach') THEN 'Pforzheim'
	WHEN City IN ('Zurich') THEN 'Zürich'
	WHEN City IN ('Asola (MN) Italien' ) THEN 'Asola'
	WHEN City IN ('Castiglion Fibocchi, Italia') THEN 'Castiglion Fibocchi'
	WHEN City IN ('Verona (VR)') THEN 'Verona'
	WHEN City IN ('Pforzheim-Mäuerach') THEN 'Pforzheim'
	WHEN City IN ('Bietigheim-Bissingen ') THEN 'Bietigheim-Bissingen'
	WHEN City IN ('Immenstadt') THEN 'Immenstadt i.Allgäu'
	WHEN City IN ('Metzingen Neuhausen') THEN 'Metzingen'
	WHEN City IN ('Leonberg-Eltingen') THEN 'Leonberg'

	
	ELSE City
END
END
