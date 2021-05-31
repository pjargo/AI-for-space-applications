__contributor__ = 'Peter Argo'
__email__ = 'peter.argo@aero.org'
__date__ = '04 April 2021'

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import datetime
import ephem
import json
from skyfield.api import EarthSatellite
from skyfield.api import wgs84
from skyfield.elementslib import osculating_elements_of
from skyfield.api import load


# Instantiate variables
LINE1 = 'line1'
LINE2 = 'line2'
LINE3 = 'line3'
# load the timescale package to input times in format needed for skyfield
ts = load.timescale()


def read_tle(*file_paths, file_type=None, return_type=None):
    """

    :param file_paths:
    :param file_type:
    :param return_type:
    :return: if return type is dictionary, return dictionary where keys are file names and values are a list of
                dictionaries representing the TLE
                e.g. {"GOES-15 data": [{'line1': tle data line 1, 'line2': tle data line 2}, ...], ...}
            if return type is None or list, return
    """
    print("file path --> ", file_paths)
    tle_data_list = list()
    first_tle = list()
    file_path: object
    if file_type == 'JSON':
        for file_path in file_paths:
            with open(file_path) as tle_file:
                return json.loads(tle_file.read())
    else:
        if return_type is None or return_type=='list':
            for file_path in file_paths:
                with open(file_path) as tle_file:
                    for idx, line in enumerate(tle_file.readlines()):
                        if idx == 0 or idx == 1:
                            print(line)
                            first_tle.append(line.strip())
                        tle_data_list.append(line.strip())
            return tle_data_list, first_tle
        else:
            tle_data_dict = dict()
            for file_path in file_paths:
                with open(file_path) as tle_file:
                    file_name = file_path.split('/')[-1].split('.')[0]
                    tle_data_list_temp = list()
                    tle_data = tle_file.readlines()
                    for idx in range(0, len(tle_data), 2):
                            tle_data_list_temp.append({
                                LINE1: check_TLE(tle_data[idx]),
                                LINE2: tle_data[idx+1]
                            })
                    tle_data_dict[file_name] = tle_data_list_temp
            return tle_data_dict


def check_TLE(line_1):
    """
    find the dates given with a space between the year and day number ie 05 13.23423 and add the missing 0
    (standard TLE dates usually have 0's instead of spaces eg. 05013.23423)
    This is condition has not been observed yet
    :param line_1:
    :return:
    """
    for i in range(20, 22):
        if line_1[i] == ' ':
            corrected_line_1 = line_1[: i] + '0' + line_1[i + 1:]
            return corrected_line_1
    return line_1


def tle_to_earth_sat(tle_data):
    """
    convert the tle dictionary to a dictionary of earth satellite objects
    :param tle_data: key = filename, value = list of TLE's as dictionaries
    :return: dictionary where key = filename, value = list of earth satellite objects
    """
    earthsat_list = list()
    earthsat_dict = dict()
    for file_name, file_data in tle_data.items():
        for tle_dict_obj in file_data:
            earthsat_list.append(EarthSatellite(tle_dict_obj[LINE1], tle_dict_obj[LINE2],
                                                name=file_name + '_' + tle_dict_obj[LINE1][18:32]))
        earthsat_dict[file_name] = earthsat_list
    return earthsat_dict


def orbital_table(tle_dict, delta_t=None, time_unit='s'):
    """
    Generate a dataframe containing ephemeris and orbital information over a time interval,
    input dict{str:list[ EarthSatellite ] }, float, output dataframe

    :param tle_dict:
    :param delta_t:
    :param time_unit:
    :return:
    """
    # dataframe columns
    eph_el_dict = {}

    # generate a time range to study the conjunction event
    epoch_list = []  # contains the final timestamps for the df
    epoch_dict = {}  # used to build the epoch list and ephemeris
    for sat in tle_dict:
        epoch_dict[sat + ' ts'] = [tle.epoch for tle in tle_dict[sat]]  # skyfield Time object list
        epoch_dict[sat + ' py'] = [t.utc_datetime() for t in epoch_dict[sat + ' ts']]  # python datetime object list
        epoch_dict[sat + ' np'] = np.array(epoch_dict[sat + ' py'])  # python datetime object numpy array

        # combine the times into a single (de-duped) sorted numpy array
        epoch_list = np.union1d(epoch_list, epoch_dict[sat + ' np'])

    # this is an option to allow propagation of the TLE's with a given
    # timestep (consider refactoring with datetime_range() )
    num_epochs_old = len(epoch_list)
    epoch_list_prop = []
    time_gap = None
    if delta_t != None:
        if time_unit.lower() == 'm':
            delta_time = datetime.timedelta(minutes=delta_t)
        elif time_unit.lower() == 'h':
            delta_time = datetime.timedelta(hours=delta_t)
        else:
            delta_time = datetime.timedelta(seconds=delta_t)

        for i in range(1, num_epochs_old):
            # if time gaps are larger than delta, propagate between
            time_gap = epoch_list[i] - epoch_list[i - 1]
            if time_gap > delta_time:
                epoch_list_prop.append(epoch_list[i - 1])
                while time_gap > delta_time:
                    epoch_list_prop.append(epoch_list_prop[-1] + delta_time)
                    time_gap = epoch_list[i] - epoch_list_prop[-1]

        if len(epoch_list_prop) > 0:
            epoch_list_prop = np.array(epoch_list_prop)

            # add in the propogated times into the epoch list
            epoch_list = np.union1d(epoch_list, epoch_list_prop)

            # time_interval = [ ts.utc( t ) for t in epoch_list ] # skyfield time version, list is in the same order as epoch list

    # store final result into list
    eph_el_dict['EPOCH'] = epoch_list

    # Generate ephemeris for the combined epoch times using the closest epochs (each tle is good for up to about 1-2 weeks)
    for sat in tle_dict:
        # prepare list for generating ephemeris
        sat_eph = []

        # find epoch nearest to given time then generate ephemeris from the corresponding TLE
        closest = None
        for i in range(len(epoch_list)):
            e0 = epoch_list[i]
            # find the index of the epoch closest to the given time
            closest = np.argmin(np.abs(epoch_dict[sat + ' np'] - e0))

            # generate the ephemeris from the tle with closest epoch
            t = ts.utc(e0)  # time_interval[ i ] # ts.utc( e0 ) but we already have this stored in time_interval
            sat_eph.append(tle_dict[sat][closest].at(t))

        # get the ra, dec, position vector
        sat_radecs = [p.radec() for p in sat_eph]

        eph_el_dict[sat + "_RA"] = [ra._degrees for ra, dec, dist in sat_radecs]  # Right Ascension
        eph_el_dict[sat + "_DEC"] = [dec._degrees for ra, dec, dist in sat_radecs]  # DEClination
        eph_el_dict[sat + "_DIST"] = [dist.km for ra, dec, dist in sat_radecs]  # Distance (from center of earth)

        # Get cooridnates for satellite positions
        # The Geocentric ICRS coordinate class has properties to convert the coordinates to other units of measure (default is AU)
        # Can also convert to other coordinate systems
        sat_coords = [p.position.km for p in sat_eph]

        eph_el_dict[sat + "_ICRS_X"] = [x for x, y, z in sat_coords]
        eph_el_dict[sat + "_ICRS_Y"] = [y for x, y, z in sat_coords]
        eph_el_dict[sat + "_ICRS_Z"] = [z for x, y, z in sat_coords]

        # Get velocities for satellite in ICRS coordinates
        # The Geocentric ICRS coordinate class has properties to convert the coordinates to other units of measure (default is AU)
        sat_velocities = [p.velocity.km_per_s for p in sat_eph]

        eph_el_dict[sat + "_ICRS_DX"] = [x for x, y, z in sat_velocities]
        eph_el_dict[sat + "_ICRS_DY"] = [y for x, y, z in sat_velocities]
        eph_el_dict[sat + "_ICRS_DZ"] = [z for x, y, z in sat_velocities]

        # Generate the Lat/Lon of the satellite
        # This uses the wgs84 model for Earth. It estimates the Lat/Lon of the point directly below the satellite.
        subpoints = [wgs84.subpoint(p) for p in sat_eph]

        eph_el_dict[sat + "_LATITUDE"] = [e.latitude.degrees for e in subpoints]
        eph_el_dict[sat + "_LONGITUDE"] = [e.longitude.degrees for e in subpoints]
        eph_el_dict[sat + "_ELEVATION"] = [e.elevation for e in subpoints]  # do e.elevation.m for meters

        # Get the osculating elements of the satellites
        ## instantiate the OsculatingElements classes for the satellites
        sat_elems = [osculating_elements_of(p) for p in sat_eph]

        # load the osculating elements information
        eph_el_dict[sat + "_APOAPSIS_DIST"] = [e.apoapsis_distance.km for e in sat_elems]
        eph_el_dict[sat + "_ARG_LAT"] = [e.argument_of_latitude.degrees for e in sat_elems]
        eph_el_dict[sat + "_ARG_PERI"] = [e.argument_of_periapsis.degrees for e in sat_elems]
        eph_el_dict[sat + "_ECC_ANOMALY"] = [e.eccentric_anomaly.degrees for e in sat_elems]
        eph_el_dict[sat + "_ECCENTRICITY"] = [e.eccentricity for e in sat_elems]
        eph_el_dict[sat + "_INCLINATION"] = [e.inclination.degrees for e in sat_elems]
        eph_el_dict[sat + "_LON_ASC_NODE"] = [e.longitude_of_ascending_node.degrees for e in sat_elems]
        eph_el_dict[sat + "_LON_PERI"] = [e.longitude_of_periapsis.degrees for e in sat_elems]
        eph_el_dict[sat + "_MEAN_ANOMALY"] = [e.mean_anomaly.degrees for e in sat_elems]
        eph_el_dict[sat + "_MEAN_LON"] = [e.mean_longitude.degrees for e in sat_elems]
        eph_el_dict[sat + "_MEAN_MOTION"] = [e.mean_motion_per_day.degrees for e in sat_elems]
        eph_el_dict[sat + "_PERI_DIST"] = [e.periapsis_distance.km for e in sat_elems]
        # eph_el_dict[ sat + "_PERI_TIME" ]           = [ e.periapsis_time for e in sat_elems ] # pandas can't interpret this object, need to cast as compatible type
        eph_el_dict[sat + "_PERIOD_DAYS"] = [e.period_in_days for e in sat_elems]
        eph_el_dict[sat + "_SEMI_LATUS_RECTUM"] = [e.semi_latus_rectum.km for e in sat_elems]
        eph_el_dict[sat + "_SEMI_MAJOR_AXIS"] = [e.semi_major_axis.km for e in sat_elems]
        eph_el_dict[sat + "_SEMI_MINOR_AXIS"] = [e.semi_minor_axis.km for e in sat_elems]
        eph_el_dict[sat + "_TRUE_ANOMALY"] = [e.true_anomaly.degrees for e in sat_elems]
        eph_el_dict[sat + "_TRUE_LON"] = [e.true_longitude.degrees for e in sat_elems]

    # convert dict into dataframe and return
    return pd.DataFrame(eph_el_dict)


def convert_epoch_to_datestring(tle_line: str):
    """
    If line 1 of 2 line elemnt, return the date_string of the epoch in the two-line
    :param tle_line: a line of a two-line elements
    :return: datestring or None
    """
    if tle_line[0] == 1 or tle_line[0] == '1':
        y_d, nbs = tle_line.split('.')[0].split(' ')[-1], tle_line.split('.')[1].split(' ')[0]

        # parse to datetime (since midnight and add the seconds) %j Day of the year as a zero-padded decimal number.
        return datetime.datetime.strptime(y_d, "%y%j") + datetime.timedelta(seconds=float("." + nbs) * 24 * 60 * 60)
    else:  # No epoch to convert in the second line of a two-line element
        return None


def reformat_date(date: str):
    return date[0:4] + '/' + date[5:6] + '/' + date[8:9]


def convert_deg_to_dec(val: str):
    deg, minute, second = val.split(':')[0], val.split(':')[1], val.split(':')[2]
    return float(deg) + float(minute)/60 + float(second)/3600


def get_lat_long_from_tle(*tle_list, tle_line0: str = 'NO SAT NAME FOUND', tle_line1: str = None,
                          tle_line2: str = None, date_offest = None):
    """
    Get the latitudes and longitude in a list of TLEs or a single TLE
    :param tle_list: list of TLEs or multiple TLEs
    :param tle_line0: The string name of the
    :param tle_line1: The string first line of the TLE
    :param tle_line2: The string second line of the TLE
    :param tle_line2: optional date offset to compute the lat and long of, number should be small
    :return: lat and long for individual TLE or last of lat and long for corresponding TLE, matched by index
    (longitude, latitude)
    """
    lat_long_list = list()
    if len(tle_list) == 0:  # Handle the individual TLE instance
        iss = ephem.readtle(tle_line0, tle_line1, tle_line2)
        compute_date = reformat_date(str(convert_epoch_to_datestring(tle_line1)))
        iss.compute(compute_date)
        if tle_line0 != 'NO SAT NAME FOUND':
            print('processed singe TLE for satellite: ', tle_line0)
        print('TLE line 1: ', tle_line1)
        if tle_line2 is not None:
            print('TLE line 2: ', tle_line2)
        print(f'longitude --> {iss.sublong} & latitude --> {iss.sublat}')
        print("\n")
        return iss.sublong, iss.sublat
    else:   # process a list of tle's
        # Check is TLE list is written as a list or multiple input strings
        if type(tle_list[0]) is list:   # If the tle_list is a list of tles, extract list
            tle_list = tle_list[0]
        if type(tle_list[0]) is dict:
            tle_list_temp = tle_list[0].get('tles')
            tle_list = list()
            for tle_lines in tle_list_temp:
                tle_list.append(tle_lines.get('line1'))
                tle_list.append(tle_lines.get('line2'))
        # Check is the list is two lines or three lines
        if tle_list[0][0] == '1':   # this is a two line TLE (without the name of the satellite)
            for idx in range(len(tle_list)):
                # print(tle_list[0][idx])
                if idx % 2 == 0:  # Continue past the first line
                    # lat_long_list.append(None)
                    continue
                else:
                    # Check is the user input the satellite name
                    if tle_line0 == 'NO SAT NAME FOUND':
                        long, lat = get_lat_long_from_tle(tle_line1=tle_list[idx - 1],
                                                          tle_line2=tle_list[idx])
                        lat_long_list.append((long/ephem.degree, lat/ephem.degree))
                    else:
                        long, lat = get_lat_long_from_tle(tle_line0=tle_line0,
                                                          tle_line1=tle_list[idx - 1],
                                                          tle_line2=tle_list[idx])
                        lat_long_list.append((convert_deg_to_dec(long), convert_deg_to_dec(lat)))
        else:   # this is a three line TLE (with the name of the satellite)
            for idx in range(len(tle_list)):
                # print(tle_list[0][idx])
                if idx+1 % 3 != 0:  # Continue past the first line
                    lat_long_list.append(None)
                    continue
                else:
                    lat_long_list.append(get_lat_long_from_tle(tle_line0=tle_list[idx-2],
                                                               tle_line1=tle_list[idx-1],
                                                               tle_line2=tle_list[idx]))
        # UPDATE TO HANDLE 3 LINE TLE'S
        return lat_long_list


if __name__ == '__main__':
    # ESTABLISH DATA FLOW FOR JSON OBJECT
    tle_dictionary = read_tle('Data/GOES-13 TLEs (neuromancer).json', file_type='JSON')
    # tle_dictionary = read_tle('iss_missions/ISS_TLES_ALL.txt', return_type='dictionary')
    earthsat_dict = tle_to_earth_sat(tle_data=tle_dictionary)

    # Running this on too large of a dataset takes too much time. Map reduce in apache spark for faster processing

    # Get a subset of the earth_sat dictionary fo the orbital table processing
    earthsat_sub_dict = {'ISS': earthsat_dict.get(list(earthsat_dict.keys())[0])[7:10]}

    iss_df = orbital_table(tle_dict=earthsat_sub_dict, delta_t=450)
    # orbital_table(tle_dict=earthsat_dict, delta_t=450)

    # tle = read_tle('Data/GOES-13 TLEs (neuromancer).json', file_type='JSON') # retun_type is list (None)
    # lat_long_list = get_lat_long_from_tle(tle)
    # plt.plot([long_lat[0] for long_lat in lat_long_list])   # Turn this into a function
    # plt.show()
    # print(*lat_long_list)
    print('end of file')
