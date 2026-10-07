param(
    [int]$PerCategory = 15,
    [int]$ThumbWidth = 1280,
    [string]$OutputRoot = $PSScriptRoot
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$api = "https://commons.wikimedia.org/w/api.php"
$userAgent = "CVDLens-Graduation-Research/2.0 (independent external evaluation dataset)"
$allowedLicense = '^(CC0|Public domain|CC BY(?:-SA)?(?: [0-9.]+)?)$'

# Twenty everyday-life strata. Queries intentionally describe concrete photographic
# scenes rather than colour names, so selection is not biased toward images that are
# known in advance to score well on a colour-confusion metric.
$categories = [ordered]@{
    road_transit       = @("urban road traffic photograph", "bus stop street photograph", "railway platform photograph", "bicycle lane street photograph", "pedestrian crossing photograph")
    signs_wayfinding   = @("directional sign photograph", "road warning sign photograph", "airport wayfinding sign photograph", "shop sign street photograph", "public information sign photograph")
    home_living        = @("living room interior photograph", "bedroom interior photograph", "home furniture photograph", "household objects table photograph", "laundry room photograph")
    kitchen_dining     = @("home kitchen photograph", "dining table meal photograph", "cookware kitchen photograph", "restaurant table photograph", "breakfast table photograph")
    grocery_retail     = @("supermarket aisle photograph", "grocery store shelves photograph", "fruit vegetable market photograph", "retail shop interior photograph", "convenience store photograph")
    food_drink         = @("mixed food plate photograph", "salad bowl photograph", "dessert cafe photograph", "beverage glasses photograph", "street food photograph")
    office_school      = @("office desk photograph", "classroom interior photograph", "school supplies photograph", "meeting room photograph", "books stationery desk photograph")
    screens_controls   = @("computer monitor desk photograph", "smartphone screen hand photograph", "control panel buttons photograph", "vending machine controls photograph", "electronic dashboard photograph")
    charts_maps        = @("color bar chart", "line chart data visualization", "thematic map colors", "transit map diagram", "weather map colors")
    tools_workshop     = @("hand tools workshop photograph", "hardware tools photograph", "workbench photograph", "factory control panel photograph", "electrical wires photograph")
    clothing_fashion   = @("colorful clothing street photograph", "clothes shop racks photograph", "shoes display photograph", "winter clothing photograph", "traditional dress photograph")
    people_social      = @("people cafe photograph", "family picnic photograph", "people walking street photograph", "friends outdoor photograph", "crowd public square photograph")
    sports_play        = @("sports field players photograph", "playground equipment photograph", "board game pieces photograph", "gym equipment photograph", "children toys photograph")
    garden_plants      = @("home garden photograph", "flower bed photograph", "vegetable garden photograph", "potted plants photograph", "autumn leaves photograph")
    landscape_water    = @("coast sea landscape photograph", "lake landscape photograph", "mountain valley photograph", "forest trail photograph", "river city photograph")
    animals_pets       = @("dog outdoors photograph", "cat indoors photograph", "bird garden photograph", "farm animals photograph", "aquarium fish photograph")
    urban_architecture = @("city street buildings photograph", "apartment exterior photograph", "public square photograph", "building entrance photograph", "parking garage photograph")
    night_lowlight     = @("night street photograph", "low light room photograph", "restaurant night interior photograph", "rainy street night photograph", "city lights evening photograph")
    weather_seasons    = @("snowy street photograph", "rain umbrella street photograph", "foggy landscape photograph", "sunny park photograph", "cloudy city photograph")
    small_color_items  = @("sewing buttons photograph", "colored pencils photograph", "medicine pills photograph", "candy pieces photograph", "game tokens photograph")
}

$imagesDir = Join-Path $OutputRoot "images"
$manifestPath = Join-Path $OutputRoot "manifest.csv"
$creditsPath = Join-Path $OutputRoot "CREDITS.md"
$summaryPath = Join-Path $OutputRoot "DATASET.md"
New-Item -ItemType Directory -Force -Path $imagesDir | Out-Null
Get-ChildItem -LiteralPath $imagesDir -File -ErrorAction SilentlyContinue | Remove-Item -Force

function PlainText([object]$metadataValue) {
    if ($null -eq $metadataValue -or $null -eq $metadataValue.value) { return "" }
    $value = [System.Net.WebUtility]::HtmlDecode([string]$metadataValue.value)
    return ([regex]::Replace($value, '<[^>]+>', '') -replace '\s+', ' ').Trim()
}

function SafeSlug([string]$value) {
    $slug = $value.ToLowerInvariant() -replace '^file:', '' -replace '[^a-z0-9]+', '-'
    return $slug.Trim('-')
}

function Get-Candidates([string]$query, [int]$offset = 0) {
    $params = @{
        action = "query"; format = "json"; formatversion = "2"
        generator = "search"; gsrnamespace = "6"; gsrlimit = "50"; gsroffset = $offset
        gsrsearch = "$query filetype:bitmap"
        prop = "imageinfo"
        iiprop = "url|size|mime|sha1|extmetadata"
        iiurlwidth = $ThumbWidth
        iiextmetadatalanguage = "en"
        iiextmetadatafilter = "LicenseShortName|LicenseUrl|Artist|Credit|AttributionRequired"
    }
    for ($attempt = 1; $attempt -le 6; $attempt++) {
        try {
            Start-Sleep -Milliseconds 750
            $response = Invoke-RestMethod -Uri $api -Method Get -Body $params -Headers @{ "User-Agent" = $userAgent }
            return @($response.query.pages)
        } catch {
            if ($attempt -eq 6) { throw }
            Start-Sleep -Seconds (5 * $attempt)
        }
    }
}

$seenSha1 = [System.Collections.Generic.HashSet[string]]::new()
$seenTitles = [System.Collections.Generic.HashSet[string]]::new()

# Make v2 independent of the original 60-image test set.
$oldManifest = Join-Path (Split-Path $PSScriptRoot -Parent) "external_eval\manifest.csv"
if (Test-Path -LiteralPath $oldManifest) {
    foreach ($old in (Import-Csv -LiteralPath $oldManifest)) {
        [void]$seenTitles.Add([string]$old.commons_title)
    }
}

$rows = [System.Collections.Generic.List[object]]::new()
foreach ($category in $categories.Keys) {
    $accepted = 0
    $queries = @($categories[$category])
    $baseQuota = [Math]::Ceiling($PerCategory / $queries.Count)
    for ($queryIndex = 0; $queryIndex -lt $queries.Count; $queryIndex++) {
        if ($accepted -ge $PerCategory) { break }
        $query = $queries[$queryIndex]
        $queryAccepted = 0
        $queryQuota = if ($queryIndex -eq $queries.Count - 1) { $PerCategory - $accepted } else { [Math]::Min($baseQuota, $PerCategory - $accepted) }
        Write-Host "[$category $accepted/$PerCategory] $query"

        foreach ($offset in @(0, 50)) {
            if ($queryAccepted -ge $queryQuota -or $accepted -ge $PerCategory) { break }
            foreach ($page in (Get-Candidates $query $offset)) {
                if ($queryAccepted -ge $queryQuota -or $accepted -ge $PerCategory) { break }
                if (-not $page.imageinfo -or -not $seenTitles.Add([string]$page.title)) { continue }
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
                if ($slug.Length -gt 48) { $slug = $slug.Substring(0, 48).Trim('-') }
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
                    sha1 = [string]$info.sha1
                    sha256 = $sha256
                    search_query = $query
                    collected_at_utc = [DateTime]::UtcNow.ToString("o")
                })
                $accepted++
                $queryAccepted++
                Write-Host "  + $filename"
            }
        }
    }
    if ($accepted -lt $PerCategory) { throw "Only collected $accepted/$PerCategory for '$category'." }
}

$rows | Export-Csv -LiteralPath $manifestPath -NoTypeInformation -Encoding UTF8
$credits = @("# External Evaluation v2 Image Credits", "", "Wikimedia Commons images collected for research evaluation. Verify each source page before redistribution.", "")
foreach ($row in $rows) {
    $author = if ($row.author) { $row.author } else { "Unknown/see source page" }
    $credits += "- **$($row.id)** - [$($row.commons_title)]($($row.source_page)); $author; [$($row.license)]($($row.license_url))"
}
Set-Content -LiteralPath $creditsPath -Value $credits -Encoding UTF8

$summary = @(
    "# Independent External Evaluation Dataset v2", "",
    "- Images: $($rows.Count)",
    "- Strata: $($categories.Count) ($PerCategory images each)",
    "- Maximum downloaded width: $ThumbWidth px", "",
    "This set is evaluation-only. Do not use it for training, checkpoint selection, threshold tuning, or post-hoc sample replacement.",
    "It excludes titles present in the original 60-image external set and records source, license, author, dimensions, SHA-1, and SHA-256 in `manifest.csv`.", "",
    "A manual relevance and safety review is required before final scoring."
)
Set-Content -LiteralPath $summaryPath -Value $summary -Encoding UTF8
Write-Host "Collected $($rows.Count) images across $($categories.Count) categories."
