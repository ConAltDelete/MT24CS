$baseUri = 'https://lmt.nibio.no/agrometbase/showweatherdata.php'
$datapath = "$($PSScriptRoot)/../../data/raw_data/nibio"

$line = Get-Content -Path "$($PSScriptRoot)/../../PRIVATE_FILES/frost_met_client.txt" -TotalCount 1
$FrostID = $line.Split(": ")[1]
$bases = @(
10 , 11 , 12 , 145 , 143 , 13 , 86 , 133 , 14 , 127 , 140 , 15 , 16 , 17 , 18 , 19 , 20 , 110 , 21 , 121 , 87 , 22 , 23 , 24 , 25 , 26 , 27 , 28 , 93 , 57 , 29 , 149 , 141 , 30 , 31 , 65 , 62 , 32 , 33 , 82 , 71 , 34 , 104 , 35 , 90 , 81 , 147 , 36 , 37 , 38 , 97 , 83 , 130 , 39 , 40 , 41 , 63 , 42 , 131 , 134 , 142 , 43 , 108 , 44 , 64 , 129 , 45 , 46 , 47 , 123 , 48 , 91 , 49 , 50 , 51 , 52 , 54 , 55 , 144 , 118 , 53 , 5 , 61 , 72
)

$station_conv = Import-csv "$($datapath)/../../info"

$jobs = @()

foreach ($base in $bases) {
	foreach ($year in 2014..2022) {
		$full_path = "$($datapath)/snowdepth_data_daily_stID_$($base).csv"
		if(Test-Path $full_path -PathType Leaf){
			continue
		}
    $jobs += Start-ThreadJob -Name "w$($base)-y$($year)" -ScriptBlock {
			param($base, $baseUri, $year, $storage,$FrostID)
			$form = @{
				sources=$base
				refrenceTime="R9/2014-03-01T00:00:00Z/2014-10-31T23:59:59Z/P1Y"
				elements="surface_snow_thickness"
				timeResolution="days"
				format="csv"
			}

			$Uri = "$($baseUri)?"
			
			foreach($el in $form.GetEnumerator()){
				$Uri += "$($el.Name)=$($el.Value)&"
			}
			
			$Uri = $Uri.Substring(0,$Uri.length-1)
			Write-Host $Uri
			curl $Uri --user $FrostID: --output $storage --retry 3 --retry-delay 5
    } -ArgumentList $base, $baseUri, $year, $full_path, $FrostID
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

