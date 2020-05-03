import requests

# TODO: doublecheck import and if there's a better way to to do this... dose requests still exist?
if __name__ == "__main__":
    train_url = "http://ufldl.stanford.edu/housenumbers/train_32x32.mat"
    test_url = "http://ufldl.stanford.edu/housenumbers/test_32x32.mat"
    extra_url = "http://ufldl.stanford.edu/housenumbers/extra_32x32.mat"

    r_train = requests.get(train_url)
    r_test = requests.get(test_url)
    r_extra = requests.get(extra_url)

    with open('./datasets/train_32x32.mat', 'wb') as d:
        d.write(r_train.content)

    with open('./datasets/test_32x32.mat', 'wb') as d:
        d.write(r_test.content)

    with open('./datasets/extra_32x32.mat', 'wb') as d:
        d.write(r_extra.content)

    # TODO: Debug mode? save to log folder? need something better than just print()
    # log metadata
    print(f"Training dataset:\n - {r_train.status_code}\n - {r_train.headers['content-type']}\n - {r_train.encoding}")
    print(f"Testing dataset:\n - {r_test.status_code}\n - {r_test.headers['content-type']}\n - {r_test.encoding}")
    print(f"Additional training dataset:\n - {r_extra.status_code}\n - {r_extra.headers['content-type']}\n - {r_extra.encoding}")
else:
    # TODO: check if this is the right exception
    raise RuntimeError