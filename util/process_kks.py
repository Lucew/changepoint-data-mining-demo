import itertools


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
    turbine = tag_groups[0][0]
    block = tag_groups[0][1]

    # get the component
    if tag_groups[1].isnumeric() or not tag_groups[2].isnumeric():
        raise ValueError(f"We could not parse {tag=}. Something with the component is off.")
    component = tag_groups[1]

    # get the measurement type
    if tag_groups[3].isnumeric() or not tag_groups[4].isnumeric():
        raise ValueError(f"We could not parse {tag=}. Something with the measurement type is off.")
    measurement_type = tag_groups[3]

    return turbine, block, component, measurement_type, sigtype


def get_info_from_list(signal_list: list[str, ...]) -> dict[str: list[str],...]:
    turbines, blocks, components, measurements, sigtypes = zip(*map(parse_kks_tag, signal_list))
    turbines = list(set(turbines))
    blocks = list(set(blocks))
    components = list(set(components))
    measurements = list(set(measurements))
    sigtypes = list(set(sigtypes))
    return {"block": blocks, "turbine": turbines, "component": components, "measurement": measurements, "Type": sigtypes}


if __name__ == "__main__":
    output = parse_kks_tag('@11BAT01CT051_XQ01')
    output2 = parse_kks_tag('11BAT01CT051_XQ01')
    print(output)
    print(output2)
