$baseUri = 'https://lmt.nibio.no/agrometbase/showweatherdata.php'
$datapath = "$($PSScriptRoot)/../../data/raw_data/nibio"

$line = Get-Content -Path "$($PSScriptRoot)/../../PRIVATE_FILES/frost_met_client.txt" -TotalCount 1
$FrostID = $line.Split(": ")[1]
$bases = @(
10 , 11 , 12 , 145 , 143 , 13 , 86 , 133 , 14 , 127 , 140 , 15 , 16 , 17 , 18 , 19 , 20 , 110 , 21 , 121 , 87 , 22 , 23 , 24 , 25 , 26 , 27 , 28 , 93 , 57 , 29 , 149 , 141 , 30 , 31 , 65 , 62 , 32 , 33 , 82 , 71 , 34 , 104 , 35 , 90 , 81 , 147 , 36 , 37 , 38 , 97 , 83 , 130 , 39 , 40 , 41 , 63 , 42 , 131 , 134 , 142 , 43 , 108 , 44 , 64 , 129 , 45 , 46 , 47 , 123 , 48 , 91 , 49 , 50 , 51 , 52 , 54 , 55 , 144 , 118 , 53 , 5 , 61 , 72
)

$jobs = @()

foreach ($base in $bases) {
	foreach ($year in 2014..2022) {
		$full_path = "$($datapath)/weather_data_raw_hour_stID$($base)_y$($year).csv"
		if(Test-Path $full_path -PathType Leaf){
			continue
		}
    $jobs += Start-ThreadJob -Name "w$($base)-y$($year)" -ScriptBlock {
			param($base, $baseUri, $year, $storage,$ID)
			$form = @{
				weatherstation=$base
				loginterval=1
				valuetype="value_raw"
				date_start="$($year)-03-01"
				date_end="$($year)-03-31"
				format="csv"
				separator="dot"
			}

			$Uri = "$($baseUri)?"

			$Uri += "weatherstation=$($form["weatherstation"])&"

			foreach ($el in @(1,297,6,7)) { # 1, 297 -< temp, nedbør
				$Uri += "elementMeasurementTypes%5B%5D=$($el)&"
			}

			foreach ($key in @("loginterval","valuetype","date_start","date_end","format","separator")) {
				$Uri += "$($key)=$($form[$key])&"
			}
			$Uri = $Uri.Substring(0,$Uri.length-1)
			Write-Host $Uri
			curl --user "$($ID):" $Uri --output $storage --retry 3 --retry-delay 5
    } -ArgumentList $base, $baseUri, $year, $full_path, $frostID
		Write-Host "Written w$($base)-y$($year)."
	}
}
if($jobs.length -eq 0) {
	Write-Host "No jobs"
} else {
	Write-Host "Downloads started..."

	Wait-Job -Job $jobs

	foreach ($job in $jobs) {
			Receive-Job -Job $job
	}
}

