// Models were: "Jurre brishti", "Muva kubwa"
// Models: "6h accumulation", "24h accumulation"
let modelName = "6h accumulation"
// Regions: Burundi, Djibouti, Eritrea, Ethiopia, Kenya, Rwanda, Somalia, South Sudan,
//          Sudan, Tanzania, Uganda, ICPAC, East Africa, All.
// There is ony one region for the Madagascar north model
let regionName = "Madagascar north";
let units = "mm/6h";			// Can be mm/h, mm/6h, mm/day, mm/week
let style = "Default";			// Can be "Default", "ICPAC", "KMD", "EMI", "ECMWF".
let plotType="Probability";		// Can be "Probability", "Values", "Mean" or "Std".
let showPercentages = true;		// On the colour scale
let maxRain = 1/24;				// Rainfall threshold in mm/h
let probability = 0.95;			// Between 0 and 1

let drawMarker = false;			// Draw the marker corresponding to the histogram location
let locationChanged = false;	// Did the lon/lat location of the histogram change?
let canvasClickX = 0;			// Where in the canvas the click was registered
let canvasClickY = 0;
let longitudeIdx = 0;			// Which location are we plotting
let latitudeIDx = 0;

// Hack to make sure clicking in the plot is handled correctly
let canvasMouseDownRects = [];

// Hack to force reloading of dates files
let dateLoadNumber = Math.floor(Math.random() * 10000);

let availableDates;				// An object containing the dates we can use
let GANForecast = [];			// An array of countsData objects

let validTimes = [];			// An array of valid times, set in updateDateMenus


// Called by the modelSelect menu
async function modelSelect() {
	modelName = document.getElementById("modelSelect").value;
	
	// Set the model description
	if (modelName == "6h accumulation") {
		document.getElementById("modelInfo").innerHTML = "The <a href=\"https://www.ecmwf.int/\" target=\"_blank\">ECMWF</a> <a href=\"https://confluence.ecmwf.int/display/FUG/Section+2+The+ECMWF+Integrated+Forecasting+System+-+IFS\" target=\"_blank\">IFS</a> output is post-processed using <a href=\"https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022MS003120\" target=\"_blank\">cGAN</a> trained on <a href=\"https://gpm.nasa.gov/data/imerg\" target=\"_blank\"> IMERG</a> v6 from 2018 and 2019 to produce forecasts of 6h rainfall intervals. Model version 1.";
	
	} else if (modelName == "24h accumulation") {
		document.getElementById("modelInfo").innerHTML = "The <a href=\"https://www.ecmwf.int/\" target=\"_blank\">ECMWF</a> <a href=\"https://confluence.ecmwf.int/display/FUG/Section+2+The+ECMWF+Integrated+Forecasting+System+-+IFS\" target=\"_blank\">IFS</a> output is post-processed using <a href=\"https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022MS003120\" target=\"_blank\">cGAN</a> trained on <a href=\"https://gpm.nasa.gov/data/imerg\" target=\"_blank\"> IMERG</a> v7 from 2018, 2019, 2020 and 2021 to produce forecasts of 24h rainfall intervals. Model version 2.";

	}
	await loadDates();		// Each model has it's own set of available dates
	await loadForecast();		// Load the currently selected forecast
	drawMarker = false;	// No longer draw the histograms
	document.getElementById("removeHistBttn").style.display = "none";	// Hide the button
	drawPlots();
}

// Called by the regionSelect menu
// function regionSelect() {
// 	regionName = document.getElementById("regionSelect").value;
// 	drawMarker = false;	// No longer draw the histograms
// 	document.getElementById("removeHistBttn").style.display = "none";	// Hide the button
// 	drawPlots();
// }

// Called by the initYearSelect, initMonthSelect,initDaySelect and initTimeSelect menus
async function initTimeSelect() {
	updateDateMenus();
	await loadForecast();
	drawPlots();
}

// Called by the validTimeSelect menu
async function validTimeSelect() {
	await loadForecast();
	drawPlots();
}

// Called by the styleSelect menu
function styleSelect() {
	style = document.getElementById("styleSelect").value;
	drawPlots();
}

// Called by the plotSelect menu
function plotSelect() {
	plotType = document.getElementById("plotSelect").value;
	if (plotType == "Probability") {
		document.getElementById("percentagesSelect").removeAttribute('disabled','');
	} else {
		document.getElementById("percentagesSelect").setAttribute('disabled','');
	}
	drawPlots();
}

// Called when the focus is lost in the value threshold input
function pValueThresholdInput() {
	drawPlots();
}

// Called by the percentagesSelect menu
function percentagesSelect() {
	if (document.getElementById("percentagesSelect").value == "Percentages") {
		showPercentages = true;
	} else {
		showPercentages = false;
	}
	drawPlots();
}

// Called when the focus is lost in the probability threshold input
function pProbabilityThresholdInput() {
	drawPlots();
}

// Called by the unitsSelect menu
function unitsSelect() {
	// Get the unitsSelect menu's value
	units = document.getElementById("unitsSelect").value;
	
	// Set the units in the description to the selected units
	document.getElementById("unitsDescription").innerHTML = units;
	
	// Update the value threshold to display in the current units
	norm = getPlotNormalisation(units);
	document.getElementById("thresholdValueSelect").value = roundSF(maxRain * norm, 3);
	
	// Draw plots with the new units
	drawPlots();
}

// Called by the showExplanations input checkbox
function showExplanation() {
	
	// If the ensemble mean or standard deviation is plotted the explanation box is always checked
	if ((document.getElementById("plotSelect").value == "Mean") ||
		(document.getElementById("plotSelect").value == "Std")) {
		
		// Check the box
		document.getElementById("showExplanation").checked = true;
		
		// Disable it
		document.getElementById("showExplanation").disabled = true;
		
	} else {
		// Enable the check box
		document.getElementById("showExplanation").disabled = false;
	}
	
	// Is the box ticked or not
	const checked = document.getElementById("showExplanation").checked;
	
	// The HTML string that the explanation will be in
	let explanationString = ``;
		
	if (checked == true) {
	
		// Get the accumulation time for the description
		let accumulationTime = 6;
		let easiestUnits = "mm/6h";
		if (document.getElementById("modelSelect").value == "24h accumulation") {
			accumulationTime = 24;
			easiestUnits = "mm/day";
		}
		// Get the value threshold for the description
		let thresholdValue = document.getElementById("thresholdValueSelect").value;
		let thresholdProbability = document.getElementById("thresholdProbabilitySelect").value;
		// Get the units for the description
		let units = document.getElementById("unitsSelect").value;
	
		// Explanation depends on the type of plot in the "Plot" menu.
		if (document.getElementById("plotSelect").value == "Probability") {
			explanationString +=
				`<b>Map description:</b> The map shows the chance that rainfall
				 accumulated over `+accumulationTime+` hours between the valid times, will
				 be above `+thresholdValue+` `+units+` at each location. This chance has
				 been calculated at each location from the histogram at that location.
				 
				 Each colour covers a range of probabilities. There are only five
				 categories to illustrate that we should not be overconfident in the
				 accuracy of our probability prediction and to make the plot clear. The
				 precise probabilities calculated are available from the histograms.
				 
				 The value threshold (`+thresholdValue+` `+units+`) can be changed to the
				 value you want in the "Value threshold" box above. The units can be set
				 in the "Units" menu also above. For example, if you are interested in
				 rainfall above 20 mm/day, first set the "Units" menu to "mm/day" and then
				 set the "Value threshold" box to "20". Setting the units to "`+
				 easiestUnits+`" means that the rainfall values plotted correspond to the
				 total rainfall over this `+accumulationTime+` hour period.
				 
				 The colour scale can be labelled as a percentage or in words by selecting
				 "Show percentages" or "Show words" in the menu above.`;
			
			if (drawMarker) {
				explanationString += ` <br><b>Histogram description:</b> The histogram
					plot to the right of each map represents the rainfall predicted by
					each ensemble member at the location marked by the cross on the map
					(at the labelled latitude and longitude). Each bar in the histogram
					shows the number of forecast ensemble members that made a prediction
					in that rainfall interval. The value threshold used in the map (`+
					thresholdValue+` `+units+`) is represented at this location by the
					blue line. The number of ensemble members to the right of the blue
					line divided by the total number of ensemble members is the predicted
					probability that the threshold will be exceeded. `+
					thresholdProbability+`% of the ensemble members are to the left of the
					red line and the rest are to the right of it. This percentage can be
					set in the "Probability threshold" box above.`;
			} else {
				explanationString += ` <br><b>Click on the map to show the histogram at that point.</b>`;
			}
								  
		} else if (document.getElementById("plotSelect").value == "Values") {
			explanationString +=
				`<b>Map description:</b> The map shows that for rainfall accumulated over
				 `+accumulationTime+` hours between the valid times, `+
				 thresholdProbability+`% of ensemble members predicted rainfall below the
				 plotted value at each location. The remaining ensemble members predicted
				 rainfall above the plotted value at each location. This rainfall value
				 has been calculated at each location from the histogram at that location.
				 
				 The probability threshold (`+thresholdProbability+`%) can be changed to
				 the percentage you want in the "Probability threshold" box above. A
				 probability threshold of 50% corresponds to the <i>ensemble median</i>
				 which is a good alternative to the ensemble mean when looking at
				 rainfall. A probability threshold of 95% indicates that only 5% of
				 ensemble members exceeded the plotted value. In that case rainfall above
				 the predicted value is unlikely. A probability threshold of 95% is a
				 good alternative to using the ensemble standard deviation to estimate the
				 range of predicted rainfall.
				 
				 Each colour covers a range of values and the precise values calculated
				 are available from the histograms. The units of the colour scale can be
				 set in the "Units" menu above. The units show the rainfall rate on
				 average over the `+accumulationTime+` hour period. Setting the units to
				 "`+easiestUnits+`" means that the rainfall values plotted correspond to
				 the total rainfall over this `+accumulationTime+` hour period.`;
			
			if (drawMarker) {
				explanationString += ` <br><b>Histogram description:</b> The histogram
					plot to the right of each map represents the rainfall predicted by
					each ensemble member at the location marked by the cross on the map
					(at the labelled latitude and longitude). Each bar in the histogram
					shows the number of forecast ensemble members that made a prediction
					in that rainfall interval. The probability threshold used in the map
					(`+thresholdProbability+`%) is represented at this location by the
					red line. `+thresholdProbability+`% of the ensemble members are to the
					left of the red line and the rest are to the right of it.
					
					The blue line corresponds to a value threshold (`+thresholdValue+` `+
					units+`). The number of ensemble members to the right of the blue line
					divided by the total number of ensemble members is the predicted
					probability that the value threshold will be exceeded. This value can
					be set in the "Value threshold" box above.`;
			} else {
				explanationString += ` <br><b>Click on the map to show the histogram at that point.</b>`;
			}
			
		} else if (document.getElementById("plotSelect").value == "Mean") {
			explanationString +=
				`<b>Map description:</b> The map shows the ensemble mean rainfall
				 accumulated over `+accumulationTime+` hours between the valid times at
				 each location.
				 
				 <span style="color:red">WARNING: The ensemble mean is usually not a good
				 summary statistic for rainfall forecasts. </span>
				 
				 The predicted rainfall distribution is far from normal (see the
				 histograms), and the ensemble mean is difficult to interpret. A good
				 alternative to the ensemble mean when looking at rainfall is the
				 <i>ensemble median</i>. The ensemble median plot can be made by selecting
				 "Values below probability" from the plot menu above and setting the
				 "Probability threshold" box to 50%.`;
			
			if (drawMarker) {
				explanationString += ` <br><b>Histogram description:</b> The histogram
					plot to the right of each map represents the rainfall predicted by
					each ensemble member at the location marked by the cross on the map
					(at the labelled latitude and longitude). Each bar in the histogram
					shows the number of forecast ensemble members that made a prediction
					in that rainfall interval.
					
					The blue line corresponds to a value threshold (`+thresholdValue+` `+
					units+`). The number of ensemble members to the right of the blue line
					divided by the total number of ensemble members is the predicted
					probability that the value threshold will be exceeded. This value can
					be set in the "Value threshold" box above. `+thresholdProbability+`%
					of the ensemble members are to the left of the red line and the rest
					are to the right of it. This percentage can be set in the "Probability
					threshold" box above.`;
			} else {
				explanationString += ` <br><b>Click on the map to show the histogram at that point.</b>`;
			}
			
		} else if (document.getElementById("plotSelect").value == "Std") {
			explanationString +=
				`<b>Map description:</b> The map shows the ensemble standard deviation of
				 rainfall accumulated over `+accumulationTime+` hours between the valid
				 times at each location.
				 
				 <span style="color:red">WARNING: The ensemble standard deviation is
				 usually not a good summary statistic for rainfall forecasts. </span>
				 
				 The predicted rainfall distribution is far from normal (see the
				 histograms), and the ensemble standard deviation is difficult to
				 interpret. A good alternative to the ensemble standard deviation when
				 looking to estimate the range of rainfall is a probability threshold of
				 95%. This plot can be made by selecting "Values below probability" from
				 the plot menu above and setting the "Probability threshold" box to 95%.`;
			
			if (drawMarker) {
				explanationString += ` <br><b>Histogram description:</b> The histogram
					plot to the right of each map represents the rainfall predicted by
					each ensemble member at the location marked by the cross on the map
					(at the labelled latitude and longitude). Each bar in the histogram
					shows the number of forecast ensemble members that made a prediction
					in that rainfall interval.
					
					The blue line corresponds to a value threshold (`+thresholdValue+` `+
					units+`). The number of ensemble members to the right of the blue line
					divided by the total number of ensemble members is the predicted
					probability that the value threshold will be exceeded. This value can
					be set in the "Value threshold" box above. `+thresholdProbability+`%
					of the ensemble members are to the left of the red line and the rest
					are to the right of it. This percentage can be set in the "Probability
					threshold" box above.`;
			} else {
				explanationString += ` <br><b>Click on the map to show the histogram at that point.</b>`;
			}
		}
		
		// Gap between paragraph and plots
		explanationString += "<br><br>";
		
		// Add the string to the paragraph
		document.getElementById("mapExplanationText").innerHTML = explanationString;
		
		// Show the explanation
		document.getElementById("mapExplanationText").style.display = "inline";
	} else {
		// Hide the explanation
		document.getElementById("mapExplanationText").style.display = "none";
	}
	
	// XXX Give the model description too
	
	
}

// Called by the removeHistBttn button
function removeHistograms() {
	drawMarker = false;	// No longer draw the histograms
	document.getElementById("removeHistBttn").style.display = "none";	// Hide the button
	drawPlots();
}

// Loads and plots the currently selected forecast
async function loadForecast() {
	let year = document.getElementById("initYearSelect").value;
	let month = document.getElementById("initMonthSelect").value;
	let day = document.getElementById("initDaySelect").value;
	let time = document.getElementById("initTimeSelect").value;
	let validTimeMenu = document.getElementById("validTimeSelect").value;
	
	// The directory name depends upon which model we are looking at
	let countsDir;
	let accumulationHours;
	if (modelName == "6h accumulation") {
		countsDir = "counts_6h";
		accumulationHours = 6;
	} else if (modelName == "24h accumulation") {
		countsDir = "counts_24h";
		accumulationHours = 24;
	}
	
	// If we should load all valid times
	if (validTimeMenu == "All") {
		for (let i=0;i<validTimes.length;i++) {
			// The cGAN forecast file to load
			let fileName = "../data/"+countsDir+"/"+year+"/counts_"+year
										 +month.padStart(2,'0')
										 +day.padStart(2,'0')
										 +"_"+time.padStart(2,'0')
										 +"_"+validTimes[i]+"h.nc";
			
			// Load data into the forecastDataObject
			await GANForecast[i].loadGANForecast(fileName, modelName, accumulationHours);
		}
	} else {	// Load a single valid time
		// The cGAN forecast file to load
		let fileName = "../data/"+countsDir+"/"+year+"/counts_"+year
									 +month.padStart(2,'0')
									 +day.padStart(2,'0')
									 +"_"+time.padStart(2,'0')
									 +"_"+validTimeMenu+"h.nc";
		
		// Load data into the forecastDataObject
		await GANForecast[0].loadGANForecast(fileName, modelName, accumulationHours);
	}
}

// Update an HTML select menu with dates that are available.
// dateObject - Can be a years, months, day or time object (from the global
//              availableDates).
// dateText   - An array containing the menu item strings to use. If empty the dateObject
//              keys are used as menu items.
// id         - The id of the select menu element in the HTML.
function updateMenu(dateObject,datesText,id) {
	
	let dates;
	if (dateObject instanceof Array) {
		// dateObject is an array of numbers. Convert it to an array of strings
		dates = new Array(dateObject.length);
		for (let i=0;i<dates.length;i++) {
			dates[i] = String(dateObject[i]);
		}
	} else {
		// dateObject's keys are a list of strings
		dates = Object.keys(dateObject);
	}
	
	// Select the HTML select menu that we are updating
	let dateSelect = document.getElementById(id);
	
	// Record the menu's value before we remove it
	let date = dateSelect.value;
	
	// Remove all of the current menu items
	while (dateSelect.hasChildNodes()) {
		dateSelect.removeChild(dateSelect.firstChild);
	}
	
	// Add the menu items specified in dates
	for (let i=0;i<dates.length;i++) {
		let option = document.createElement("option");
		option.value = dates[i];
		if (datesText.length > 0) {
			option.innerHTML = datesText[i];
		} else {
			option.innerHTML = dates[i];
		}
		dateSelect.appendChild(option);
	}
	
	// Add the "Plot all valid times" menu item.
	if (datesText.length > dateObject.length) {
		if (datesText[datesText.length-1] == "Plot all valid times") {
			let option = document.createElement("option");
			option.value = "All";
			option.innerHTML = "Plot all valid times";
			dateSelect.appendChild(option);
		}
	}
	
	if (date != "All") {
		// If the specified year/month/day/time/valid time does not exist.
		if (!(dates.includes(date))) {
			date = dates[dates.length-1];	// Pick the final one
		}
	}
	
	// Set the menu to the value it should be
	dateSelect.value = date;
	
	// Return the value set
	return date;
}

function updateDateMenus() {
	
	// The available months are listed in availableDates
	year = updateMenu(availableDates,[],"initYearSelect");
	
	// The available months depend upon the year
	let yearObject = availableDates[String(year)];
	month = updateMenu(yearObject,[],"initMonthSelect");
	
	// The available days depend upon the year and month
	let monthObject = yearObject[String(month)];
	day = updateMenu(monthObject,[],"initDaySelect");
	
	// The available times depend upon the year, month and day
	let daysObject = monthObject[String(day)];
	// We use a custom string for the time menu elements
	let times = Object.keys(daysObject);
	let timeStrings = new Array(times.length);
	for (let i=0;i<times.length;i++) {
		timeStrings[i] = times[i].padStart(2,'0')+":00 UTC";
	}
	time = updateMenu(daysObject,timeStrings,"initTimeSelect");
	
	// The available valid times depend upon the year, month, day and time.
	validTimes = daysObject[String(time)];	// validTimes is an Array
	// We use a custom string for the valid time menu elements
	let validTimeStrings = new Array(validTimes.length+1);
	for (let i=0;i<validTimes.length;i++) {
		// What's the valid date? (YYYY-MM-DD)
		validDate = timeOffsetToDate(validTimes[i]+parseInt(time),
									 year+"-"+String(month).padStart(2,'0')
										 +"-"+String(day).padStart(2,'0'));
				
		validTimeStrings[i] = validDate.getUTCFullYear()
							+"-"+String(validDate.getUTCMonth()+1).padStart(2,'0')
							+"-"+String(validDate.getUTCDate()).padStart(2,'0')
							+" "+String(validDate.getUTCHours()).padStart(2,'0')
							+":00 UTC (+"+validTimes[i]+"h)";
	}
	// Add an "Plot all valid times" option
	validTimeStrings[validTimes.length] = "Plot all valid times";
	updateMenu(validTimes,validTimeStrings,"validTimeSelect");
}

async function loadDates() {
	// Fetch a remote file
	let fileName;
	if (modelName == "6h accumulation") {
		fileName = "../data/counts_6h/available_dates.json?"+dateLoadNumber;
	} else if (modelName == "24h accumulation") {
		fileName = "../data/counts_24h/available_dates.json?"+dateLoadNumber;
	}
	// dateLoadNumber ensures that the available_dates.json file is not cached
	dateLoadNumber += 1;
	if (dateLoadNumber > 10000) {
		dateLoadNumber = 0;
	}
	const response = await fetch(fileName);
	
	// Parse the JSON arrayBuffer of the file and return the resulting object
	availableDates = await response.json();
	
	// Pick the final date to load
// 	let years = Object.keys(availableDates);
// 	let year = years[years.length-1];
// 	let yearObject = availableDates[year];
// 	let months = Object.keys(yearObject);
// 	let month = months[months.length-1];
// 	let monthObject = yearObject[month];
// 	let days = Object.keys(monthObject);
// 	let day = days[days.length-1];
// 	let daysObject = monthObject[day];
// 	let times = Object.keys(daysObject);
// 	let time = times[times.length-1];
// 	let validTimes = daysObject[time];
// 	let validTime = validTimes[validTimes.length-1];
	
	// Set the menus to match the loaded dates
	// Probably doesn't do anything. Overridden by updateDateMenus.
// 	document.getElementById("initYearSelect").value = year;
// 	document.getElementById("initMonthSelect").value = month;
// 	document.getElementById("initDaySelect").value = day;
// 	document.getElementById("initTimeSelect").value = time;
// 	document.getElementById("validTimeSelect").value = String(validTime);
	
	updateDateMenus();
	
	// By default plot all lead times
// 	document.getElementById("validTimeSelect").value = "All";
}

function initControls() {
	// XXX Actually should keep the settings on reload and reload the correct plots
	
	// Following the HTML elements in order
	
	document.getElementById("modelSelect").value = modelName;
	
	// document.getElementById("regionSelect").value = regionName;
	
	document.getElementById("styleSelect").value = style;
	
	document.getElementById("plotSelect").value = plotType;
	
	// Need to get the units correct
	norm = getPlotNormalisation(units);
	document.getElementById("thresholdValueSelect").value = roundSF(maxRain * norm, 3);
	
	document.getElementById("unitsDescription").innerHTML = units;
	
	if (showPercentages) {
		document.getElementById("percentagesSelect").value = "Percentages";
	} else {
		document.getElementById("percentagesSelect").value = "Words";
	}
	
	if (plotType == "Probability") {
		document.getElementById("percentagesSelect").removeAttribute('disabled','');
	} else {
		document.getElementById("percentagesSelect").setAttribute('disabled','');
	}
	
	document.getElementById("thresholdProbabilitySelect").value = roundSF((probability*100), 4);
	
	document.getElementById("unitsSelect").value = units;
}

// Function to inform the user what is going on
//    code - 0 = Not waiting
//           1 = Waiting for data to load
//           2 = Waiting for calculations
//           3 = Waiting for plots to draw
// message - A description of what we are waiting for
function showLoadingStatus(code, message) {

	if (code == 0) {	// We are not waiting
		document.getElementById("statusText").style.color = "black";
		
	} else {			// We are waiting
		document.getElementById("statusText").style.color = "#cc0000";	// dark red
	}
	
	// Inform the user what is going on
	document.getElementById("statusText").innerHTML = message;
}

async function init() {

	// Set the default values of the plot controls
	initControls();
	
	// Specify the function to call to inform the user what is going on
	setStatusUpdateFunction(showLoadingStatus);
	
	// GANForecast is a global array of countsData objects
	// XXX Make according to validTimes.length
	for (let i=0;i<7;i++) {
		GANForecast[i] = new countsData();		// Create a countsData object
	}
	
	// If the region names are not loaded yet, then load them and wait for them to be loaded
	// First argument is the directory, second argument is the file name.
	await GANForecast[0].regionSpec.loadRegionNames("../boundaries", "regional_names.json");
	// All point at the same regionSpec
	// XXX Make according to validTimes.length
	for (let i=1;i<7;i++) {
		GANForecast[i].regionSpec = GANForecast[0].regionSpec;
	}
	
	// Load a list of the available forecasts
	await loadDates();
	
	// Load the currently selected forecast
	await loadForecast();
	
	// Draw everything
	await drawPlots();
	
	// Stop the default window scroll when the arrow keys are pressed.
	// XXX Probably should attach this to the relevant canvas instead, to enable arrow
	// keys in the input boxes. See:
	// https://stackoverflow.com/questions/8916620/disable-arrow-key-scrolling-in-users-browser
	window.addEventListener("keydown", function(e) {
    	if(["ArrowUp","ArrowDown","ArrowLeft","ArrowRight"].indexOf(e.code) > -1) {
       		e.preventDefault();
    	}
	}, false);
	
	// Detect if the enter or return key is pressed in the document
	document.addEventListener("keydown", function(event) {
		
		if (event.key === "Enter") {
			drawPlots();
		
		} else if (event.key === "ArrowRight") {
			// If we are currently drawing the location marker
			if (drawMarker) {
				// Increase the longitude index by one
				longitudeIdx += 1;
				
				// The lon/lat location changes with a mouse click
				locationChanged = false;
			
				// Draw the plots
				requestAnimationFrame(drawPlots);
			}
			
		} else if (event.key === "ArrowLeft") {
			// If we are currently drawing the location marker
			if (drawMarker) {
				// Increase the longitude index by one
				longitudeIdx -= 1;
				
				// The lon/lat location changes with a mouse click
				locationChanged = false;
			
				// Draw the plots
				requestAnimationFrame(drawPlots);
			}
		
		} else if (event.key === "ArrowUp") {
			// If we are currently drawing the location marker
			if (drawMarker) {
				// Increase the longitude index by one
				latitudeIdx += 1;
				
				// The lon/lat location changes with a mouse click
				locationChanged = false;
			
				// Draw the plots
				requestAnimationFrame(drawPlots);
			}
		
		} else if (event.key === "ArrowDown") {
			// If we are currently drawing the location marker
			if (drawMarker) {
				// Increase the longitude index by one
				latitudeIdx -= 1;
				
				// The lon/lat location changes with a mouse click
				locationChanged = false;
			
				// Draw the plots
				requestAnimationFrame(drawPlots);
			}
		}
	});
}

// Listens for the mouse in the supplied rectangle (corresponding to the plot picture)
function listenForMouse(canvasNum) {
	
	// Get canvas for events
	const canvas = document.getElementById("mapCanvas"+canvasNum);
	
	// Detect the mouse location when it is within the canvas element
	canvas.addEventListener('mousedown', function(evt) {
		
		// Get the current rectangle for dealing with plot clicks from a global variable
		plotRect = canvasMouseDownRects[canvasNum];
		
		// Get the mouse position in the canvas element
		let canvasRect = canvas.getBoundingClientRect();
		let clickX = evt.clientX - canvasRect.left;
		let clickY = evt.clientY - canvasRect.top;
		
		// Get the mouse location within the plot image boundary
		let xMouse = Math.floor(clickX) - Math.round(plotRect[0]);
		let yMouse = Math.floor(clickY) - Math.round(plotRect[1]);
		
		// Width and height of the plot rectangle
		let width = Math.round(plotRect[2]-plotRect[0]);
		let height = Math.round(plotRect[3]-plotRect[1]);
		
		// If the mouse is within the plot rectangle
		if (xMouse>=0 && yMouse>=0 && xMouse<width && yMouse<height) {
			
			// Save the click location for other functions
			canvasClickX = clickX;
			canvasClickY = clickY;
			
			// When the plots are drawn, also draw the marker
			drawMarker = true;
			document.getElementById("removeHistBttn").style.display = "inline";	// Show the button
			
			// The lon/lat location changes with a mouse click
			locationChanged = true;
		
			// Draw the plots
			requestAnimationFrame(drawPlots);
		}
	});
}

async function drawPlots() {

	// It's easier to update the plot explanation every time the plots are drawn
	showExplanation();
	
	// See what the input boxes say
	let norm = getPlotNormalisation(units);
	maxRain = document.getElementById("thresholdValueSelect").value / norm;
	probability = document.getElementById("thresholdProbabilitySelect").value / 100.0;
	
	// Find out how many plots to make
	let plotAllValidTimes = document.getElementById("validTimeSelect").value;
	if (plotAllValidTimes == "All") {
		numCanvases = validTimes.length;
	} else {
		numCanvases = 1;
	}
	
	// Ensure the correct number of map canvases exist
	let canvasNum=0;
	while (document.getElementById("mapCanvas"+canvasNum) != null) {
		// If this canvas is not needed
		if (canvasNum+1 > numCanvases) {
			canvasElement = document.getElementById("mapCanvas"+canvasNum);
			canvasElement.remove();
		}
		canvasNum += 1;
	}
	// If there are insufficient canvases
	while (canvasNum < numCanvases) {
		const canvasElement = document.createElement("canvas");
		canvasElement.id = "mapCanvas"+canvasNum;
		canvasElement.width = 513;
		canvasElement.height = 504;
		canvasElement.innerHTML = "Your browser does not support the HTML canvas tag.";
		// canvasElement.style="border:1px solid grey";
		document.body.appendChild(canvasElement);
		// Listen for clicks in the canvas
		listenForMouse(canvasNum);
		canvasNum += 1;
	}
	
	// Create the necessary histogram canvases (Same code as above almost)
	if (drawMarker) {
		// Ensure the correct number of histogram canvases exist
		let canvasNum=0;
		while (document.getElementById("histogramCanvas"+canvasNum) != null) {
			// If this canvas is not needed
			if (canvasNum+1 > numCanvases) {
				canvasElement = document.getElementById("histogramCanvas"+canvasNum);
				canvasElement.remove();
			}
			canvasNum += 1;
		}
		// If there are insufficient canvases
		brIdx=0;	// Keep track of the number of line breaks
		while (canvasNum < numCanvases) {
			const canvasElement = document.createElement("canvas");
			canvasElement.id = "histogramCanvas"+canvasNum;
			canvasElement.width = 511;
			canvasElement.height = 504;
			canvasElement.innerHTML = "Your browser does not support the HTML canvas tag.";
			// canvasElement.style="border:1px solid grey";
			
			// Place the histogram canvas just after the map canvas
			const mapElement = document.getElementById("mapCanvas"+canvasNum);
			mapElement.insertAdjacentElement("afterend", canvasElement);
			
			// If the map drawn is to the right of the histogram
			mapRect = mapElement.getClientRects();
			histogramRect = canvasElement.getClientRects();
			// With a buffer to account for funky layout engines
			if (mapRect[0].x > histogramRect[0].x + mapElement.width/2) {
				// Insert <br> before the map
				const brElement = document.createElement("br");
				brElement.id = "mapNewLine"+brIdx;
				brIdx += 1;
				mapElement.insertAdjacentElement("beforebegin", brElement);
			}
						
			canvasNum += 1;
		}
	} else {
		// Remove all histogram canvases
		let canvasNum=0;
		while (document.getElementById("histogramCanvas"+canvasNum) != null) {
			canvasElement = document.getElementById("histogramCanvas"+canvasNum);
			canvasElement.remove();
			canvasNum += 1;
		}
		
		// Remove all mapNewLine line breaks
		let brIdx=0;
		while (document.getElementById("mapNewLine"+brIdx) != null) {
			brElement = document.getElementById("mapNewLine"+brIdx);
			brElement.remove();
			brIdx += 1;
		}
	}
	
	// Reset the array of rectangles containing map canvas boundaries
	canvasMouseDownRects = [];
	
	// Draw plots in each canvas
	for (let canvasNum=0;canvasNum<numCanvases;canvasNum++) {
		
		// Get the context for plotting
		const canvas = document.getElementById("mapCanvas"+canvasNum);
		const ctx = canvas.getContext("2d");
		
		// Erase the canvas
		ctx.clearRect(0,0,canvas.width,canvas.height);
		
		let x = 2, y=2;			// Location of plot from top left
		let width = 500;		// Width of plot in pixels
		let height = 500;		// Height of plot in pixels
		
		// Must use await unless all of the region shape data is loaded in advance
		// otherwise the region shape data could be loaded multiple times and corrupted.
		
		// The rectangles within which the plots are drawn
		let plotRect;
		if (plotType == "Probability") {
			plotRect = await GANForecast[canvasNum].plotExceedanceProbability(ctx, x, y, width, height,
																   maxRain, units, style,
																   showPercentages, regionName);
		} else if (plotType == "Values") {
			plotRect = await GANForecast[canvasNum].plotExceedanceValue(ctx, x, y, width, height,
															 probability, units, style, regionName);
		} else if (plotType == "Mean") {
			plotRect = await GANForecast[canvasNum].plotMean(ctx, x, y, width, height,
															 units, style, regionName);
		} else if (plotType == "Std") {
			plotRect = await GANForecast[canvasNum].plotStd(ctx, x, y, width, height,
															units, style, regionName);
		}
		
		// Save plotRect for understanding map clicks
		canvasMouseDownRects[canvasMouseDownRects.length] = plotRect;
		
		// Plot the marker and associated histogram
		if (drawMarker) {
		
			// Need the longitude range in the current plot
			let [minLatIdx,maxLatIdx,minLonIdx,maxLonIdx] = GANForecast[0].computeLatLonIdxBounds(regionName);
			
			// If the location has changed (set the latitude and longitude indices)
			if (locationChanged) {
			
				// Get the mouse location within the plot image boundary
				let xMouse = Math.floor(canvasClickX) - Math.round(plotRect[0]);
				let yMouse = Math.floor(canvasClickY) - Math.round(plotRect[1]);
				
				// Find the corresponding latitude and longitude indices
				longitudeIdx = minLonIdx + Math.round(xMouse * (maxLonIdx-minLonIdx)
															 / (plotRect[2]-plotRect[0]));
				latitudeIdx = maxLatIdx - Math.round(yMouse * (maxLatIdx-minLatIdx)
															/ (plotRect[3]-plotRect[1]));
				
				// By default the location hasn't changed so set this flag now
				locationChanged = false;
			
			} else {	// The location has not changed (set click location from the lat/lon indices)
				canvasClickX = (longitudeIdx - minLonIdx) * (plotRect[2]-plotRect[0])
														  / (maxLonIdx-minLonIdx) + plotRect[0];
				canvasClickY = (maxLatIdx - latitudeIdx) * (plotRect[3]-plotRect[1])
														 / (maxLatIdx-minLatIdx) + plotRect[1];
			}
			
			let markerWidth = 10;	// Width of the plot marker in pixels
			
			// Thick black cross
			ctx.beginPath();
			ctx.strokeStyle = "#000000";
			ctx.lineWidth = 3;
			ctx.moveTo(canvasClickX - markerWidth/2, canvasClickY - markerWidth/2);
			ctx.lineTo(canvasClickX + markerWidth/2, canvasClickY + markerWidth/2);
			ctx.moveTo(canvasClickX + markerWidth/2, canvasClickY - markerWidth/2);
			ctx.lineTo(canvasClickX - markerWidth/2, canvasClickY + markerWidth/2);
			ctx.stroke();
			
			// Thin white cross
			ctx.beginPath();
			ctx.strokeStyle = "#ffffff";
			ctx.lineWidth = 1;
			ctx.moveTo(canvasClickX - markerWidth/2, canvasClickY - markerWidth/2);
			ctx.lineTo(canvasClickX + markerWidth/2, canvasClickY + markerWidth/2);
			ctx.moveTo(canvasClickX + markerWidth/2, canvasClickY - markerWidth/2);
			ctx.lineTo(canvasClickX - markerWidth/2, canvasClickY + markerWidth/2);
			ctx.stroke();
		
			// Put a line dividing the two plots
			ctx.strokeStyle = "#b6b6b6";
			ctx.lineWidth = 1;
			ctx.beginPath();
			ctx.moveTo(512, 25);
			ctx.lineTo(512, 504-25);
			ctx.stroke();
			
			// Create a new histogram specification
			let barChartSpec = new barChartSpecification();
			
			let y2 = y;
			let x2 = 8;				// Change the location of the second plot
			
			// Get the context for plotting
			const histogramCanvas = document.getElementById("histogramCanvas"+canvasNum);
			const histogramCtx = histogramCanvas.getContext("2d");
			
			// Erase the canvas
			histogramCtx.clearRect(0,0,histogramCanvas.width,histogramCanvas.height);
			
			// Plot the histogram and wait for it to finish
			await GANForecast[canvasNum].plotHistogram(histogramCtx, x2, y2, width, height,
						maxRain, probability,latitudeIdx, longitudeIdx, units, barChartSpec);
		}
	}
}

init();
