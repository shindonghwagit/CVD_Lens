param(
    [int]$PerCategory = 10,
    [int]$ThumbWidth = 1280,
    [string]$OutputRoot = (Join-Path $PSScriptRoot ".")
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$api = "https://commons.wikimedia.org/w/api.php"
$userAgent = "CVDLens-Graduation-Research/1.0 (external evaluation dataset)"
$allowedLicense = '^(CC0|Public domain|CC BY(?:-SA)?(?: [0-9.]+)?)$'
$categories = [ordered]@{
    traffic = @("traffic signal intersection", "road signs street", "metro transit map", "airport direction signs", "pedestrian crossing signal")
    charts_ui = @("colored bar chart", "line chart statistics", "thematic map colors", "computer interface screenshot", "metro map diagram", "data visualization chart", "pie chart statistics")
    food = @("red green vegetables", "colorful fruit market", "food plate restaurant", "supermarket produce")
    nature = @("red flowers green grass", "blue sky landscape", "sunflower field landscape", "blue sea coast")
    indoor_lowlight = @("night street lights", "indoor room lighting", "neon signs night", "dark restaurant interior")
    people_clothing = @("portrait colorful clothing", "children colorful clothes", "traditional colorful costume", "street fashion colorful clothes", "festival traditional dress")
}
$excludedTitles = [System.Collections.Generic.HashSet[string]]::new([string[]]@(
    "File:YUL Condos from Rue Guy intersection, Montreal.jpg",
    "File:Metro Vancouver Transit Police Ford CVPI.jpg",
    "File:Ursa Major2.jpg",
    "File:Jackson Liquors New Orleans 1951.jpg",
    "File:IBM Thinkpad R51.jpg",
    "File:Blond woman in a pink underwear on a field with yellow flowers 02.jpg",
    "File:Andean Man bleach bypass.jpg",
    "File:Charlotte catherine de la Trémoille de Condé Guillain Louvre LP 400.JPG"
))

$imagesDir = Join-Path $OutputRoot "images"
$manifestPath = Join-Path $OutputRoot "manifest.csv"
$creditsPath = Join-Path $OutputRoot "CREDITS.md"
New-Item -ItemType Directory -Force -Path $imagesDir | Out-Null
Get-ChildItem -LiteralPath $imagesDir -File -ErrorAction SilentlyContinue | Remove-Item -Force

function PlainText([object]$metadataValue) {
    if ($null -eq $metadataValue -or $null -eq $metadataValue.value) { return "" }
    $text = [System.Net.WebUtility]::HtmlDecode([string]$metadataValue.value)
    return ([regex]::Replace($text, '<[^>]+>', '') -replace '\s+', ' ').Trim()
}

function SafeSlug([string]$value) {
    $slug = $value.ToLowerInvariant() -replace '^file:', '' -replace '[^a-z0-9]+', '-'
    return $slug.Trim('-')
}

function Get-Candidates([string]$query) {
    $params = @{
        action = "query"; format = "json"; formatversion = "2"
        generator = "search"; gsrnamespace = "6"; gsrlimit = "40"
        gsrsearch = "$query filetype:bitmap"
        prop = "imageinfo"
        iiprop = "url|size|mime|sha1|extmetadata"
        iiurlwidth = $ThumbWidth
        iiextmetadatalanguage = "en"
        iiextmetadatafilter = "LicenseShortName|LicenseUrl|Artist|Credit|AttributionRequired"
    }
    for ($attempt = 1; $attempt -le 6; $attempt++) {
        try {
            Start-Sleep -Seconds 3
            $response = Invoke-RestMethod -Uri $api -Method Get -Body $params -Headers @{ "User-Agent" = $userAgent }
            return @($response.query.pages)
        } catch {
            if ($attempt -eq 6) { throw }
            $delay = 15 * $attempt
            Write-Warning "Commons API request failed; retrying in $delay seconds ($attempt/6)."
            Start-Sleep -Seconds $delay
        }
    }
}

$rows = [System.Collections.Generic.List[object]]::new()
$seenSha1 = [System.Collections.Generic.HashSet[string]]::new()
$seenTitles = [System.Collections.Generic.HashSet[string]]::new()

foreach ($category in $categories.Keys) {
    $accepted = 0
    $queries = @($categories[$category])
    $baseQuota = [Math]::Ceiling($PerCategory / $queries.Count)
    for ($queryIndex = 0; $queryIndex -lt $queries.Count; $queryIndex++) {
        $query = $queries[$queryIndex]
        if ($accepted -ge $PerCategory) { break }
        $queryAccepted = 0
        $queryQuota = if ($queryIndex -eq $queries.Count - 1) { $PerCategory - $accepted } else { $baseQuota }
        Write-Host "[$category] searching: $query"
        foreach ($page in (Get-Candidates $query)) {
            if ($accepted -ge $PerCategory -or $queryAccepted -ge $queryQuota) { break }
            if (-not $page.imageinfo -or $excludedTitles.Contains([string]$page.title) -or -not $seenTitles.Add([string]$page.title)) { continue }
            $info = $page.imageinfo[0]
            $meta = $info.extmetadata
            $license = PlainText $meta.LicenseShortName
            if ($license -notmatch $allowedLicense) { continue }
            if ($info.width -lt 640 -or $info.height -lt 480) { continue }
            if ($info.mime -notin @("image/jpeg", "image/png", "image/webp")) { continue }
            if (-not $seenSha1.Add([string]$info.sha1)) { continue }

            $extension = switch ($info.mime) {
                "image/png" { ".png" }
                "image/webp" { ".webp" }
                default { ".jpg" }
            }
            $number = $accepted + 1
            $slug = SafeSlug ([string]$page.title)
            if ($slug.Length -gt 45) { $slug = $slug.Substring(0, 45).Trim('-') }
            $filename = "{0}_{1:D2}_{2}{3}" -f $category, $number, $slug, $extension
            $destination = Join-Path $imagesDir $filename
            $downloadUrl = if ($info.thumburl) { $info.thumburl } else { $info.url }

            try {
                Invoke-WebRequest -Uri $downloadUrl -OutFile $destination -Headers @{ "User-Agent" = $userAgent }
                $file = Get-Item -LiteralPath $destination
                if ($file.Length -lt 20KB) { Remove-Item -LiteralPath $destination; continue }
                $sha256 = (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLowerInvariant()
            } catch {
                if (Test-Path -LiteralPath $destination) { Remove-Item -LiteralPath $destination }
                Write-Warning "download failed: $($page.title)"
                continue
            }

            $rows.Add([pscustomobject]@{
                id = ("{0}_{1:D2}" -f $category, $number)
                category = $category
                filename = $filename
                commons_title = [string]$page.title
                source_page = [string]$info.descriptionurl
                original_url = [string]$info.url
                download_url = [string]$downloadUrl
                license = $license
                license_url = PlainText $meta.LicenseUrl
                author = PlainText $meta.Artist
                credit = PlainText $meta.Credit
                attribution_required = PlainText $meta.AttributionRequired
                original_width = [int]$info.width
                original_height = [int]$info.height
                downloaded_bytes = [long]$file.Length
                sha256 = $sha256
                search_query = $query
                collected_at_utc = [DateTime]::UtcNow.ToString("o")
            })
            $accepted++
            $queryAccepted++
            Write-Host "  + $filename ($license)"
            Start-Sleep -Milliseconds 150
        }
    }
    if ($accepted -lt $PerCategory) {
        throw "Only collected $accepted/$PerCategory images for category '$category'."
    }
}

$rows | Export-Csv -LiteralPath $manifestPath -NoTypeInformation -Encoding UTF8

$credits = @(
    "# External Evaluation Image Credits"
    ""
    "Images were collected from Wikimedia Commons for research evaluation."
    "Follow each linked file page and license when redistributing an image."
    ""
)
foreach ($row in $rows) {
    $author = if ($row.author) { $row.author } else { "Unknown/see source page" }
    $credits += "- **$($row.id)** — [$($row.commons_title)]($($row.source_page)); $author; [$($row.license)]($($row.license_url))"
}
Set-Content -LiteralPath $creditsPath -Value $credits -Encoding UTF8

Write-Host ""
Write-Host "Collected $($rows.Count) images."
Write-Host "Manifest: $manifestPath"
Write-Host "Credits:  $creditsPath"
