import numpy as np


def validate_numeric_range(floaty: float|int|np.ndarray[int|float], min: float|int = -np.inf, \
                           max: float|int = np.inf, message: str = "Funny message for value error") -> None:
    """
    Validates an int, float, or a numpy array of floats or ints.
    Args:
        floaty: int, float, or a numpy array of floats or ints. Whatever you want to validate buddy.
        min: The minimum value inclusive.
        max: The max value inclusive.
        message: The error message. Not acceptable if not funny. In the future need validation to make sure the message is funny-> Eric responsible.
    Raises:
        ValueError: If the value or any of the values in the array is not within the boundaries.
    :return: jack shit.
    """
    # Next line is active in future.
    # validate_error_message(message)
    try:  # handling ndarrays.
        if not (np.all(min <= floaty) and np.all(floaty <= max)):
            raise ValueError(message)
    except TypeError:  # handling regular ints or floats.
        if not min <= floaty <= max:
            raise ValueError(message)


def validate_error_message(message: str):
    """
    Validates error messages.

    Args:
        message: The error message.

    Raises:
        ValueError: If the writer of the message had the nerve to try to be serious.
    """
    funny = True
    humor_validation = None  # placeholder until Eric creates a validation for the messages.
    if message != humor_validation:
        # funny = False
        pass
    if not funny:
        raise ValueError("Absolutely unacceptable! This country was built on standards and you try sneaking by an unfunny error message? \
                         We can be very understanding. But at some things are unforgiveable.")


def validate_par_j(par_j: float|np.ndarray) -> None:
    """
    Validates the parameter j.

    Args:
        par_j: The parameter j.

    Raises:
        ValueError: If the parameter j is not between 0 and 360.
    """
    validate_numeric_range(floaty=par_j, min=0, max=360, message="This is the day of the year in degrees. It must be between 0 and 360.")


def validate_day(day: int|np.ndarray) -> None:
    """
    Validates the day.

    Args:
        day: day number up to 366.

    Raises:
        ValueError: If the day is not between 1 and 366.
    """
    validate_numeric_range(floaty=day, min=1, max=366, message="New concepts for years should first be approved in cenate. We work with 366 days max. "
                                                         "(day not between 1 and 366)")


def validate_sun_declination(sun_declination: float|np.ndarray) -> None:
    """
    Validates the sun declination value.

    Args:
        sun_declination: The sun declination value.

    Raises:
        ValueError: If the sun declination is not between -23.5 and 23.5 degrees. The real value is -23.44 and 23.44,
        but we want to allow some wiggle room for rounding errors and shit like that.
    """
    validate_numeric_range(floaty=sun_declination, min=-23.5, max=23.5, message="This calculation is for Earth. Not your magical planet. "
                         "(Sun declination not between -23.44 and 23.44)")


def validate_time_parameter(time_parameter: float|np.ndarray) -> None:
    """
    Validates the time parameter value.

    Args:
        time_parameter: The time parameter value.

    Raises:
        ValueError: If the time parameter is not between -1 and 1.
    """
    validate_numeric_range(floaty=time_parameter, min=-1, max=1, message="How did you even do that? (time parameter not between -1 and 1)")


def validate_loc_time(loco_time: float|np.ndarray) -> None:
    """
    Validates the local time.

    Args:
        loco_time: The local time.

    Raises:
        ValueError: If the local time is not between 0 and 1.
    """
    validate_numeric_range(floaty=loco_time, min=0, max=1, message="This ain't the time for jokes buddy! (local time not between 0 and 1)")


def validate_timezone(timezone: float|np.ndarray) -> None:
    """
    Validates the timezone.

    Args:
        timezone: The timezone.

    Raises:
        ValueError: If the timezone is not between -1 and 1.
    """
    validate_numeric_range(floaty=timezone, min=-1, max=1, message="This ain't the timezone for jokes buddy! (timezone not between -1 and 1)")


def validate_longitude(longitude: float|np.ndarray) -> None:
    """
    Validates the longitude.

    Args:
        longitude: The longitude.

    Raises:
        ValueError: If the longitude is not between -180 and 180.
    """
    validate_numeric_range(floaty=longitude, min=-180, max=180, message="Wrong length buddy. Size matters! (longitude not between -180 and 180)")


def validate_latitude(latitude: float|np.ndarray) -> None:
    """
    Validates the latitude.

    Args:
        latitude: The latitude.

    Raises:
        ValueError: If the latitude is not between -90 and 90.
    """
    validate_numeric_range(floaty=latitude, min=-90, max=90, message="Wrong latitude buddy. You are aware this is earth, right? (latitude not between -90 and 90)")


def validate_hour_angle(hour_angley: float|np.ndarray) -> None:
    """
    Validates the hour angle.

    Args:
        hour_angley: The hour angle.

    Raises:
        ValueError: If the hour_angle is not between --360 and 360.
    """
    validate_numeric_range(floaty=hour_angley, min=-360, max=360, message="It's just the angle and GSP is huge! (hour angle not between -360 and 360)")


def validate_mean_loc_time(mean_loco_time: float|np.ndarray) -> None:
    """
    Validates the mean local time.

    Args:
        mean_loco_time: The mean local time.

    Raises:
        ValueError: If the mean local time is not between 0 and 1.
    """
    validate_numeric_range(floaty=mean_loco_time, min=0, max=1, message="This ain't the time for jokes buddy! (mean local time not between 0 and 1)")


def validate_true_loc_time(true_loco_time: float|np.ndarray) -> None:
    """
    Validates the true local time.

    Args:
        true_loco_time: The true local time.

    Raises:
        ValueError: If the true local time is not between 0 and 1.
    """
    validate_numeric_range(floaty=true_loco_time, min=0, max=1, message="Wrong time for jokes buckaroo! (true local time not between 0 and 1)")


def validate_angle_incidence(angle_incidence: float|np.ndarray) -> None:
    """
    Validates the angle of incidence.

    Args:
        angle_incidence: The angle of incidence in degrees.

    Raises:
        ValueError: If the angle of incidence is not between 0 and 180 degrees.
    """
    validate_numeric_range(floaty=angle_incidence, min=0, max=180, message="What kind of angles are you working with? (angle of incidence not between 0 and 180")


def validate_sol_altitude(sol_altitude: float|np.ndarray) -> None:
    """
    Validates the solar altitude.

    Args:
        sol_altitude: The solar altitude in degrees.

    Raises:
        ValueError: If the solar altitude is not between .90 and 90.
    """
    validate_numeric_range(floaty=sol_altitude, min=-90, max=90, message="I don't think it works like that bro/sis! (solar altitude not between -90 and 90)")


def validate_sol_azimuth(sol_azimuth: float|np.ndarray) -> None:
    """
    Validates the solar azimuth value.

    Args:
        sol_azimuth: The solar azimuth value.

    Raises:
        ValueError: If the solar azimuth is not between -360 and 360.
    """
    validate_numeric_range(floaty=sol_azimuth, min=-360, max=360, message="I don't think it works like that bro/sis! (sun azimuth not between -360 and 360)")


def validate_module_angle(module_angle: float|int|np.ndarray) -> None:
    """
    Validates the module angle value.

    Args:
        module_angle: The module angle value.

    Raises:
        ValueError: If the module angle is not between 0 and 90.
    """
    validate_numeric_range(floaty=module_angle, min=0, max=90, message="What the hell are you doing with your modules? Facing away from \
                           the sun is a questionable choice, partner. (module angle not between 0 and 90)")


def validate_module_azimuth(module_azimuth: float|int|np.ndarray) -> None:
    """
    Validates the module azimuth value.

    Args:
        module_azimuth: The module azimuth value.

    Raises:
        ValueError: If the module azimuth is not between -360 and 360. You coult argue it should be between 0 and 360,
        but some systems say west is minus 90 and south minus 180.
    """
    validate_numeric_range(floaty=module_azimuth, min=-360, max=360, message="What kind of tricks are you pulling here? (module azimuth not between -360 and 360)")


def validate_direct_radiation(direct_radiation: float|int|np.ndarray) -> None:
    """
    Validates the direct radiation value.

    Args:
        direct_radiation: total radiation on the perpendicular to the sun in W/m².

    Raises:
        ValueError: If the direct radiation is negative.
    """
    validate_numeric_range(floaty=direct_radiation, min=0, message="Is the sun stealing energy, dawg? (negative perpendicular direct radiation)")


def validate_diffuse_radiation(diffuse_radiation: float|int|np.ndarray) -> None:
    """
    Validates the diffuse radiation value.

    Args:
        diffuse_radiation: diffuse radiation in W/m².

    Raises:
        ValueError: If the diffuse radiation is negative.
    """
    validate_numeric_range(floaty=diffuse_radiation, min=0, message="Greedy environment is stealing our energy? (negative diffuse radiation)")


def validate_global_radiation(global_radiation: float|int|np.ndarray) -> None:
    """
    Validates the global radiation value.

    Args:
        global_radiation: global radiation = direct + diffus. (W/m²).

    Raises:
        ValueError: If the global radiation is negative.
    """
    validate_numeric_range(floaty=global_radiation, min=0, message="Is the sun stealing energy, dawg? (negative global radiation)")


def validate_albedo_value(albedo_value: float|int|np.ndarray) -> None:
    """
    Validates the global albedo value.

    Args:
        albedo_value: The albedo value, usually 0.2.

    Raises:
        ValueError: If the albedo value is not between 0 and 1.
    """
    validate_numeric_range(floaty=albedo_value, min=0, max=1, message="The share of the diffuse radiation that is reflection radiation is usually not negative "
                         "or over 100 percent. Either you just discovered a flaw in our understanding of physics, "
                         "or this is a mistake. (Albedo value not between 0 and 1)")
