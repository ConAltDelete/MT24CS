
$line = Get-Content -Path "$($PSScriptRoot)/../../PRIVATE_FILES/frost_met_client.txt" -TotalCount 1
$FrostID = $line.Split(": ")[1]
$frosturi = "https://frost.met.no/sources/v0.jsonld?types=SensorSystem&geometry=nearest(POINT(%20))"
$frosturi2 = "https://frost.met.no/observations/v0.csv?"

$datapath = "$($PSScriptRoot)/../../data/info"
$datafile = "$($datapath)/StationIDInfo_snow.csv"
$stationlist = @(
10,11,12,145,143,13,86,133,14,127,140,15,16,17,18,19,20,110,21,121,87,22,23,24,25,26,27,28,93,57,29,149,141,30,31,65,62,32,33,82,71,34,104,35,90,81,147,36,37,38,97,83,130,39,40,41,63,42,131,134,142,43,108,44,64,129,45,46,47,123,48,91,49,50,51,52,54,55,144,118,53,5,61,72
)



$attributes = @(
	"ID","Name","Method","Long","Lati","FrostName","ErrorDist","S0","D0","S1","D1","S2","D2","S3","D3","S4","D4"
)

New-Item -Path $datafile -Value "$($attributes -join ";")`n" -Force

foreach($id in $stationlist){
foreach($weatherel in @("surface_snow_thickness","mean(surface_snow_thickness)","max(surface_snow_thickness)")){
	$webreq = Invoke-WebRequest -Uri "https://lmt.nibio.no/services/rest/weatherstation/getstation?weatherStationId=$($id)" | ConvertFrom-Json
	Add-Content -Path $datafile -Value (@($webreq.weatherStationId,$webreq.name,$webreq.latitude,$webreq.longitude) -join ";") -NoNewline -Encoding "UTF8"
	$frostlocal = curl "https://frost.met.no/sources/v0.jsonld?types=SensorSystem&geometry=nearest(POINT($($webreq.longitude)%20$($webreq.latitude)))" -u "$($FrostID):" | ConvertFrom-Json
	$frostdata = curl "https://frost.met.no/sources/v0.jsonld?types=SensorSystem&elements=$($weatherel)&geometry=nearest(POINT($($webreq.longitude)%20$($webreq.latitude)))&nearestmaxcount=5&validtime=2014-03-01/2022-10-31" -u "$($FrostID):" | ConvertFrom-Json

	Add-Content -Path $datafile -Value ";$($frostlocal.data.id);$($weatherel);$($frostlocal.data.distance)" -NoNewline
	foreach($i in 0..4){
		$substat = $frostdata.data[$i]
		Add-Content -Path $datafile -Value ";$(@($substat.id, $substat.distance) -join ";")" -NoNewline
	}
	Add-Content -Path $datafile -Value "`n" -NoNewline
	$j = 1
	do {
			Write-Host "Attempting: id $($id) on index $($j)"
			$weatherdata = curl "https://frost.met.no/observations/v0.csv?sources=$($frostdata.data[$j].id)&referencetime=R9/2014-03-01T00:00:00Z/2014-10-31T23:59:59Z/P1Y&elements=$($weatherel)" -u "$($FrostID):"
			$fileoutput = "$($datapath)/../raw_data/MET/SnowDepthV_$($weatherel)_To_$($id)_FROM_$($frostdata.data[$j].id).csv"
		try {
			$weatherdata = $weatherdata | ConvertFrom-Json
			Write-Host "$($weatherdata."@type")"
			Write-Host "Did not find for id $($id) at index $($j)"
			$DebugStations = curl "https://frost.met.no/sources/v0.jsonld?types=SensorSystem&geometry=nearest(POINT($($webreq.longitude)%20$($webreq.latitude)))&nearestmaxcount=5" -u "$($FrostID):" | ConvertFrom-Json
			curl "https://frost.met.no/observations/availableTimeSeries/v0.jsonld?sources=$($DebugStations.data.id -join "%2C")" -u "$($FrostID):" --output "$($datapath)/Debug/DebugStationFailure_nr$($id).json"
			if($weatherdata."@type" -eq "ErrorResponse"){
			} else {
				Write-Host "Found for id $($id) at index $($j)?"
				Add-Content -Path $fileoutput -Value $weatherdata
			}
		} catch {
			Write-Host "Found for id $($id) at index $($j)"
			Add-Content -Path $fileoutput -Value $weatherdata -Force
		}
		$j = $j + 1
	} while($j -le ([math]::Min(5,$frostdata.data.Length) - 1))
}}

