package org.flag4j.arrays.sparse.sparse_tensor;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.sparse.CooTensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CooTensorTransposeTests {
    static CooTensor A;
    static Shape aShape;
    static double[] aEntries;
    static int[][] aIndices;

    static CooTensor exp;
    static Shape expShape;
    static double[] expEntries;
    static int[][] expIndices;

    @Test
    void transposeTests() {
        // ----------------------- sub-case 1 -----------------------
        aShape = new Shape(3, 4, 2, 1, 5);
        aEntries = new double[]{0.3369876012099134, -1.2778944176165061, 0.3003526634585191, -2.871698370528301, -0.7476330328637798, 0.0005862743350540067, 0.27050784380653264, -0.8804057494085602, 0.3433759561302188, -0.6271337465977211, -0.535101554634142, -1.9109729935630098, 0.7218896402785697, 0.8404528180001959, 0.5203214826154133, 0.3371173022767966, -0.6726729311708143, 0.08353784066932919};
        aIndices = new int[][]{
                {0, 0, 1, 0, 1},
                {0, 1, 1, 0, 4},
                {0, 2, 0, 0, 0},
                {0, 2, 1, 0, 1},
                {0, 3, 0, 0, 2},
                {1, 1, 0, 0, 0},
                {1, 1, 1, 0, 0},
                {1, 1, 1, 0, 2},
                {1, 2, 1, 0, 0},
                {1, 2, 1, 0, 4},
                {2, 0, 0, 0, 0},
                {2, 0, 0, 0, 4},
                {2, 0, 1, 0, 4},
                {2, 1, 0, 0, 0},
                {2, 1, 1, 0, 1},
                {2, 1, 1, 0, 3},
                {2, 1, 1, 0, 4},
                {2, 3, 0, 0, 2}};
        A = new CooTensor(aShape, aEntries, aIndices);

        expShape = new Shape(5, 4, 2, 1, 3);
        expEntries = new double[]{-0.535101554634142, 0.0005862743350540067, 0.8404528180001959, 0.27050784380653264, 0.3003526634585191, 0.3433759561302188, 0.3369876012099134, 0.5203214826154133, -2.871698370528301, -0.8804057494085602, -0.7476330328637798, 0.08353784066932919, 0.3371173022767966, -1.9109729935630098, 0.7218896402785697, -1.2778944176165061, -0.6726729311708143, -0.6271337465977211};
        expIndices = new int[][]{
                {0, 0, 0, 0, 2},
                {0, 1, 0, 0, 1},
                {0, 1, 0, 0, 2},
                {0, 1, 1, 0, 1},
                {0, 2, 0, 0, 0},
                {0, 2, 1, 0, 1},
                {1, 0, 1, 0, 0},
                {1, 1, 1, 0, 2},
                {1, 2, 1, 0, 0},
                {2, 1, 1, 0, 1},
                {2, 3, 0, 0, 0},
                {2, 3, 0, 0, 2},
                {3, 1, 1, 0, 2},
                {4, 0, 0, 0, 2},
                {4, 0, 1, 0, 2},
                {4, 1, 1, 0, 0},
                {4, 1, 1, 0, 2},
                {4, 2, 1, 0, 1}};
        exp = new CooTensor(expShape, expEntries, expIndices);
        assertEquals(exp, A.T());

        // ----------------------- sub-case 2 -----------------------
        aShape = new Shape(3, 4, 2, 1, 5);
        aEntries = new double[]{0.5495928325292689, -0.4063964781150276, -2.059856306563159, 0.21626315719045072, 0.700470594502127, -0.9551549318514719, 0.20625241662218816, -0.002902831572102285, 0.027379206391192006, -0.048618618464467335, -0.062183293526625626};
        aIndices = new int[][]{
                {0, 1, 0, 0, 1},
                {0, 1, 0, 0, 2},
                {0, 1, 1, 0, 3},
                {1, 0, 1, 0, 0},
                {1, 1, 0, 0, 0},
                {1, 2, 0, 0, 1},
                {2, 0, 1, 0, 1},
                {2, 0, 1, 0, 3},
                {2, 0, 1, 0, 4},
                {2, 1, 1, 0, 3},
                {2, 2, 1, 0, 1}};
        A = new CooTensor(aShape, aEntries, aIndices);

        expShape = new Shape(2, 4, 3, 1, 5);
        expEntries = new double[]{0.5495928325292689, -0.4063964781150276, 0.700470594502127, -0.9551549318514719, 0.21626315719045072, 0.20625241662218816, -0.002902831572102285, 0.027379206391192006, -2.059856306563159, -0.048618618464467335, -0.062183293526625626};
        expIndices = new int[][]{
                {0, 1, 0, 0, 1},
                {0, 1, 0, 0, 2},
                {0, 1, 1, 0, 0},
                {0, 2, 1, 0, 1},
                {1, 0, 1, 0, 0},
                {1, 0, 2, 0, 1},
                {1, 0, 2, 0, 3},
                {1, 0, 2, 0, 4},
                {1, 1, 0, 0, 3},
                {1, 1, 2, 0, 3},
                {1, 2, 2, 0, 1}};
        exp = new CooTensor(expShape, expEntries, expIndices);
        assertEquals(exp, A.T(0, 2));
        assertEquals(exp, A.T(2, 0));

        // ----------------------- sub-case 3 -----------------------
        aShape = new Shape(3, 4, 2, 1, 5);
        aEntries = new double[]{0.7755080687938627, -0.35048451266628083, -1.1870851252844012, 0.7738863613063384, 1.486547192285357, 0.2252740863565129, -0.2975058226231608, -0.09472050340155494, -0.15340207810250436, 2.84248590850813, -1.7279407634814916};
        aIndices = new int[][]{
                {0, 1, 0, 0, 2},
                {0, 1, 0, 0, 3},
                {0, 1, 1, 0, 2},
                {0, 3, 1, 0, 1},
                {1, 0, 0, 0, 2},
                {1, 1, 1, 0, 0},
                {2, 0, 0, 0, 0},
                {2, 0, 0, 0, 3},
                {2, 3, 0, 0, 0},
                {2, 3, 0, 0, 1},
                {2, 3, 1, 0, 0}};
        A = new CooTensor(aShape, aEntries, aIndices);

        expShape = new Shape(3, 1, 5, 4, 2);
        expEntries = new double[]{0.7738863613063384, 0.7755080687938627, -1.1870851252844012, -0.35048451266628083, 0.2252740863565129, 1.486547192285357, -0.2975058226231608, -0.15340207810250436, -1.7279407634814916, 2.84248590850813, -0.09472050340155494};
        expIndices = new int[][]{
                {0, 0, 1, 3, 1},
                {0, 0, 2, 1, 0},
                {0, 0, 2, 1, 1},
                {0, 0, 3, 1, 0},
                {1, 0, 0, 1, 1},
                {1, 0, 2, 0, 0},
                {2, 0, 0, 0, 0},
                {2, 0, 0, 3, 0},
                {2, 0, 0, 3, 1},
                {2, 0, 1, 3, 0},
                {2, 0, 3, 0, 0}};
        exp = new CooTensor(expShape, expEntries, expIndices);
        assertEquals(exp, A.T(0, 3, 4, 1, 2));

        // ----------------------- sub-case 4 -----------------------
        assertThrows(IllegalArgumentException.class, ()->A.T(0, 1, 3, 2));
        assertThrows(IllegalArgumentException.class, ()->A.T(0, 3, 4, 1, 2, 5));
        assertThrows(IllegalArgumentException.class, ()->A.T(0, 3, -4, 1, 2));
        assertThrows(IllegalArgumentException.class, ()->A.T(0, 15, 4, 1, 2));
        assertThrows(IndexOutOfBoundsException.class, ()->A.T(5, 1));
    }
}
