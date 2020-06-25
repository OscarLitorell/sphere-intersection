
import math
import numpy as np

class Sphere:

    def __init__(self, radius, center):
        self.radius = radius
        self.center = center


    def sphere_intersection(self, other):
        r = np.linalg.norm(self.center - other.center)

        a = self.radius
        b = other.radius

        p = 0.5 * ((a ** 2 - b ** 2) / r ** 2 + 1)

        circle_center = self.center * (1 - p) + other.center * p
        circle_radius = math.sqrt(a ** 2 - (p * r) ** 2)
        normal = self.center - other.center

        return Circle(circle_center, circle_radius, normal)


class Circle:

    def __init__(self, center, radius, normal):
        self.center = center
        self.radius = radius
        self.normal = normal


    def get_plane(self):
        w = self.normal.dot(self.center)
        return Plane(*self.normal, w)



class Plane:

    def __init__(self, n_x, n_y, n_z, w):
        "Create a plane with the form n_x*n + n_y*y + n_z*z = w, where n is the normal vector."
        # TODO: n_x, n_y, n_z -> normal as np.array([n_x, n_y, n_z])
        self.n = np.array([n_x, n_y, n_z])
        self.w = w


    def normalize(self):
        mag = np.linalg.norm(self.n)
        self.n /= mag
        self.w /= mag

    @staticmethod
    def from_points(points):
        normal = np.cross(points[1] - points[0], points[2] - points[0])
        normal /= np.linalg.norm(normal)
        w = normal.dot(points[0])
        return Plane(*normal, w)


    def intersect_plane(self, other):
        
        # System matrix
        m = np.vstack((self.n, other.n))

        # Solution vector
        s = np.array([[self.w], [other.w]])

        # For a line: (p_0 and p_1 are points on the line)
        # l = p_0 + (p_1 - p_0) * t

        index = 0

        for i in range(3):
            if np.all(m[:,i] == 0):
                index = i
                break

        new_m = np.delete(m, index, 1) # p_0[i] = 0
        removed = m[:, index].reshape((2, 1))

        inv = np.linalg.inv(new_m)

        p_0 = inv.dot(s).reshape((2,))
        p_1 = inv.dot(s - removed) # p_1[i] = 1


        p_0 = np.insert(p_0, index, 0)
        p_1 = np.insert(p_1, index, 1)
        
        d_p = p_1 - p_0 # delta p
        d_p /= np.linalg.norm(d_p)

        return Line(p_0, d_p)
    

    def intersect_line(self, line):
        self.normalize()

        point = self.w / line.step.dot(self.n) * line.step

        projected_start = line.start - np.dot(line.start, self.n) * self.n
        point += projected_start

        return point



class Line:

    def __init__(self, start, step):
        self.start = start
        self.step = step


def spheres_intersection(spheres):
    circle1 = spheres[0].sphere_intersection(spheres[1])
    circle2 = spheres[1].sphere_intersection(spheres[2])
    plane1 = circle1.get_plane()
    plane2 = circle2.get_plane()
    line = plane1.intersect_plane(plane2)

    midpoint = Plane.from_points([sphere.center for sphere in spheres]).intersect_line(line)

    dist = math.sqrt(circle1.radius ** 2 - np.linalg.norm(midpoint - circle1.center) ** 2)

    p_1 = midpoint + line.step * dist
    p_2 = midpoint - line.step * dist
    
    return p_1, p_2


def test():
    sensors = [np.random.random([3]) * 20 - 10 for _ in range(3)]
    point = np.random.random([3]) * 20 - 10

    spheres = [Sphere(np.linalg.norm(sensor - point), sensor) for sensor in sensors]

    print(point)
    intersections = spheres_intersection(spheres)

    print(intersections)
    print([np.linalg.norm(intersection - point) for intersection in intersections])
    




def main():
    test()

    spheres = [
        Sphere(5.0, np.array([0, 0, 0])),
        Sphere(5.0, np.array([1.0, 0, 0])),
        Sphere(5.0, np.array([0, 1.0, 0]))
    ]

    spheres_intersection(spheres)



if __name__ == "__main__":
    main()
