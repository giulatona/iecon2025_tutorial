# Data Information

This directory contains the UCI Individual Household Electric Power Consumption dataset.

## Dataset Details

- **Source**: UCI Machine Learning Repository
- **URL**: https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption
- **Period**: December 2006 - November 2010 (47 months)
- **Frequency**: 1-minute intervals
- **Total Records**: 2,075,259 measurements
- **File Size**: ~127 MB

## Variables

1. **Date**: Date in format dd/mm/yyyy
2. **Time**: Time in format hh:mm:ss
3. **Global_active_power**: Household global minute-averaged active power (kilowatts)
4. **Global_reactive_power**: Household global minute-averaged reactive power (kilowatts)
5. **Voltage**: Minute-averaged voltage (volts)
6. **Global_intensity**: Household global minute-averaged current intensity (amperes)
7. **Sub_metering_1**: Energy sub-metering No. 1 (Watt-hour) - Kitchen (dishwasher, oven, microwave)
8. **Sub_metering_2**: Energy sub-metering No. 2 (Watt-hour) - Laundry room (washing machine, tumble-dryer, light)
9. **Sub_metering_3**: Energy sub-metering No. 3 (Watt-hour) - Electric water-heater and air-conditioner

## Notes

- Missing values are represented with '?'
- All measurements taken every minute
- Global_active_power = Sub_metering_1 + Sub_metering_2 + Sub_metering_3 + (unmeasured active energy)
- File format: CSV with semicolon (;) as separator

## Citation

If using this dataset, please cite:
Georges Hebrail, Alice Berard. Individual household electric power consumption Data Set. UCI Machine Learning Repository, 2012.