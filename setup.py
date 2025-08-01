from setuptools import find_packages, setup

package_name = 'awsim_to_kitti'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='arka',
    maintainer_email='arka@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'awsim_to_kitti = awsim_to_kitti.awsim_to_kitti_dataset:main',
        ],
    },
)
