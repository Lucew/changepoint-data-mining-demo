import itertools
import typing


def parse_kks_tag(tag: str) -> [str, str, str, str, str]:

    # check for the offset (if we have an @)
    if tag[0] == "@":
        tag = tag[1:]

    # split the sigtype
    sigsplit = tag.split("_", 1)
    if len(sigsplit) != 2:
        raise ValueError(f"We could not parse {tag=}. There is no signal type (e.g., XQ01, ZQ01).")
    tag, sigtype = sigsplit

    # parse the tag in groups
    tag_groups = ["".join(grp) for _, grp in itertools.groupby(tag, lambda s: s.isdigit())]

    # check that our assumption is correct
    if not len(tag_groups) >= 5:
        raise ValueError(f"We could not parse {tag=}. Could not extract mimimum five groups.")

    # check the first group
    if len(tag_groups[0]) != 2 or not tag_groups[0].isnumeric():
        raise ValueError(f"We could not parse {tag=}. Something with block and turbine is off.")

    # get the turbine and block
    block = tag_groups[0][0]
    turbine = tag_groups[0][1]

    # get the component
    if tag_groups[1].isnumeric() or not tag_groups[2].isnumeric():
        raise ValueError(f"We could not parse {tag=}. Something with the component is off.")
    component = tag_groups[1]

    # get the measurement type
    if tag_groups[3].isnumeric() or not tag_groups[4].isnumeric():
        raise ValueError(f"We could not parse {tag=}. Something with the measurement type is off.")
    measurement_type = tag_groups[3]

    return block, turbine, component, measurement_type, sigtype


def get_info_from_list(signal_list: list[str], unique: bool = True) -> dict[str: list[str]]:
    blocks, turbines, components, measurements, sigtypes = zip(*map(parse_kks_tag, signal_list))

    # make lists
    turbines = list(turbines)
    blocks = list(blocks)
    components = list(components)
    measurements = list(measurements)
    sigtypes = list(sigtypes)

    if unique:
        turbines = list(set(turbines))
        blocks = list(set(blocks))
        components = list(set(components))
        measurements = list(set(measurements))
        sigtypes = list(set(sigtypes))

    return {"block": blocks, "turbine": turbines, "component": components, "measurement": measurements, "type": sigtypes}


def signal_name_mask(signal_names: list[str],
                     block_list: typing.Iterable[str] = None,
                     turbine_list: typing.Iterable[str] = None,
                     component_list: typing.Iterable[str] = None,
                     measurement_list: typing.Iterable[str] = None,
                     type_list: typing.Iterable[str] = None) -> list[bool]:

    # transform into sets
    block_list = set(block_list) if block_list is not None else None
    turbine_list = set(turbine_list) if turbine_list is not None else None
    component_list = set(component_list) if component_list is not None else None
    measurement_list = set(measurement_list) if measurement_list is not None else None
    type_list = set(type_list) if type_list is not None else None

    result_list = [
        len(parsed_tag := parse_kks_tag(signal_name))
        and (block_list is None or parsed_tag[0] in block_list)
        and (turbine_list is None or parsed_tag[1] in turbine_list)
        and (component_list is None or parsed_tag[2] in component_list)
        and (measurement_list is None or parsed_tag[3] in measurement_list)
        and (type_list is None or parsed_tag[4] in type_list)
        for signal_name in signal_names
    ]
    return result_list


def signal_name_filter(signal_names: list[str],
                       block_list: typing.Iterable[str] = None,
                       turbine_list: typing.Iterable[str] = None,
                       component_list: typing.Iterable[str] = None,
                       measurement_list: typing.Iterable[str] = None,
                       type_list: typing.Iterable[str] = None) -> list[str]:

    # get the filter mask
    mask = signal_name_mask(signal_names,
                            block_list=block_list,
                            turbine_list=turbine_list,
                            component_list=component_list,
                            measurement_list=measurement_list,
                            type_list=type_list)

    # filter the actual name list
    filtered_signal_names = list(itertools.compress(signal_names, mask))
    return filtered_signal_names


if __name__ == "__main__":
    output = parse_kks_tag('@11BAT01CT051_XQ01')
    output2 = parse_kks_tag('11BAT01CT051_XQ01')
    print(output)
    print(output2)
