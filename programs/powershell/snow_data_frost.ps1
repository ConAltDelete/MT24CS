
$line = Get-Content -Path "$($PSScriptRoot)/../../PRIVATE_FILES/frost_met_client.txt" -TotalCount 1
$FrostID = $line.Split(": ")[1]
$frosturi = "https://frost.met.no/sources/v0.jsonld?types=SensorSystem&geometry=nearest(POINT(%20))"
$frosturi2 = "https://frost.met.no/observations/v0.csv?"

$datapath = "$($PSScriptRoot)/../../data/info"
$datafile = "$($datapath)/StationIDInfo_snow.csv"
$stationlist = @(
10,11,12,145,143,13,86,133,14,127,140,15,16,17,18,19,20,110,21,121,87,22,23,24,25,26,27,28,93,57,29,149,141,30,31,65,62,32,33,82,71,34,104,35,90,81,147,36,37,38,97,83,130,39,40,41,63,42,131,134,142,43,108,44,64,129,45,46,47,123,48,91,49,50,51,52,54,55,144,118,53,5,61,72
)

$weather_element = "surface_snow_thickness"

foreach($id in $stationlist){
	$webreq = Invoke-WebRequest -Uri "https://lmt.nibio.no/services/rest/weatherstation/getstation?weatherStationId=$($id)" | ConvertFrom-Json
	$frostlocal = curl "https://frost.met.no/sources/v0.jsonld?types=SensorSystem&geometry=nearest(POINT($($webreq.longitude)%20$($webreq.latitude)))&validtime=2014-03-01/2022-10-31&elements=$($weather_element)" -u "$($FrostID):" | ConvertFrom-Json
	# Write-Host $frostlocal
	$relevant_id = $frostlocal.data[0].id

	$weatherdata = curl "https://frost.met.no/observations/v0.csv?sources=$($relevant_id)&referencetime=R9/2014-03-01T00:00:00Z/2014-10-31T23:59:59Z/P1Y&elements=$($weather_element)" -u "$($FrostID):" --output "$($PSScriptRoot)/../../data/raw_data/weatherdata_snow_stID$($id).csv"
	Add-Content -Path "$($datapath)/snow_station_distance_info.csv" -Value "$($id);$($frostlocal.distance)" -Force

}

