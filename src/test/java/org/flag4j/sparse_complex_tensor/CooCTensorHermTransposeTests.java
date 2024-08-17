package org.flag4j.sparse_complex_tensor;

import org.flag4j.arrays_old.sparse.CooCTensor;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.core.Shape;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CooCTensorHermTransposeTests {
    static CooCTensor A;
    static Shape aShape;
    static CNumber[] aEntries;
    static int[][] aIndices;

    static CooCTensor exp;
    static Shape expShape;
    static CNumber[] expEntries;
    static int[][] expIndices;

    @Test
    void hermTransposeTests() {
        // ----------------------- Sub-case 1 -----------------------
        aShape = new Shape(3, 4, 2, 1, 5);
        aEntries = new CNumber[]{new CNumber(0.6522, 0.3289), new CNumber(0.7167, 0.9757), new CNumber(0.7091, 0.0283), new CNumber(0.2897, 0.9174), new CNumber(0.2596, 0.4461), new CNumber(0.4028, 0.9296), new CNumber(0.9347, 0.0967), new CNumber(0.4156, 0.7123), new CNumber(0.5299, 0.2536), new CNumber(0.8344, 0.3449), new CNumber(0.3802, 0.6804)};
        aIndices = new int[][]{
                {0, 0, 0, 0, 2},
                {0, 2, 0, 0, 0},
                {1, 0, 0, 0, 3},
                {1, 1, 0, 0, 1},
                {1, 1, 1, 0, 3},
                {1, 3, 0, 0, 3},
                {1, 3, 1, 0, 4},
                {2, 1, 0, 0, 2},
                {2, 1, 0, 0, 3},
                {2, 1, 1, 0, 3},
                {2, 2, 1, 0, 3}};
        A = new CooCTensor(aShape, aEntries, aIndices);

        expShape = new Shape(5, 4, 2, 1, 3);
        expEntries = new CNumber[]{new CNumber(0.7167, -0.9757), new CNumber(0.2897, -0.9174), new CNumber(0.6522, -0.3289), new CNumber(0.4156, -0.7123), new CNumber(0.7091, -0.0283), new CNumber(0.5299, -0.2536), new CNumber(0.2596, -0.4461), new CNumber(0.8344, -0.3449), new CNumber(0.3802, -0.6804), new CNumber(0.4028, -0.9296), new CNumber(0.9347, -0.0967)};
        expIndices = new int[][]{
                {0, 2, 0, 0, 0},
                {1, 1, 0, 0, 1},
                {2, 0, 0, 0, 0},
                {2, 1, 0, 0, 2},
                {3, 0, 0, 0, 1},
                {3, 1, 0, 0, 2},
                {3, 1, 1, 0, 1},
                {3, 1, 1, 0, 2},
                {3, 2, 1, 0, 2},
                {3, 3, 0, 0, 1},
                {4, 3, 1, 0, 1}};
        exp = new CooCTensor(expShape, expEntries, expIndices);
        assertEquals(exp, A.H());

        // ----------------------- Sub-case 2 -----------------------
        aShape = new Shape(3, 4, 2, 1, 5);
        aEntries = new CNumber[]{new CNumber(0.8546, 0.6631), new CNumber(0.8133, 0.2484), new CNumber(0.0033, 0.7366), new CNumber(0.0844, 0.3648), new CNumber(0.8295, 0.2401), new CNumber(0.1182, 0.7329), new CNumber(0.6894, 0.0494), new CNumber(0.4388, 0.4951), new CNumber(0.8198, 0.6859), new CNumber(0.8987, 0.6718), new CNumber(0.2785, 0.8425)};
        aIndices = new int[][]{
                {0, 2, 0, 0, 1},
                {0, 2, 1, 0, 0},
                {0, 3, 1, 0, 3},
                {1, 0, 0, 0, 3},
                {1, 0, 1, 0, 2},
                {1, 2, 1, 0, 4},
                {1, 3, 1, 0, 3},
                {2, 0, 1, 0, 1},
                {2, 0, 1, 0, 4},
                {2, 1, 1, 0, 0},
                {2, 1, 1, 0, 2}};
        A = new CooCTensor(aShape, aEntries, aIndices);

        expShape = new Shape(2, 4, 3, 1, 5);
        expEntries = new CNumber[]{new CNumber(0.0844, -0.3648), new CNumber(0.8546, -0.6631), new CNumber(0.8295, -0.2401), new CNumber(0.4388, -0.4951), new CNumber(0.8198, -0.6859), new CNumber(0.8987, -0.6718), new CNumber(0.2785, -0.8425), new CNumber(0.8133, -0.2484), new CNumber(0.1182, -0.7329), new CNumber(0.0033, -0.7366), new CNumber(0.6894, -0.0494)};
        expIndices = new int[][]{
                {0, 0, 1, 0, 3},
                {0, 2, 0, 0, 1},
                {1, 0, 1, 0, 2},
                {1, 0, 2, 0, 1},
                {1, 0, 2, 0, 4},
                {1, 1, 2, 0, 0},
                {1, 1, 2, 0, 2},
                {1, 2, 0, 0, 0},
                {1, 2, 1, 0, 4},
                {1, 3, 0, 0, 3},
                {1, 3, 1, 0, 3}};
        exp = new CooCTensor(expShape, expEntries, expIndices);

        assertEquals(exp, A.H(0, 2));
        assertEquals(exp, A.H(2, 0));

        // ----------------------- Sub-case 3 -----------------------
        aShape = new Shape(3, 4, 2, 1, 5);
        aEntries = new CNumber[]{new CNumber(0.2252, 0.8303), new CNumber(0.7022, 0.9162), new CNumber(0.6672, 0.2974), new CNumber(0.4421, 0.7193), new CNumber(0.3796, 0.9056), new CNumber(0.2784, 0.3588), new CNumber(0.3264, 0.909), new CNumber(0.1959, 0.2546), new CNumber(0.7772, 0.396), new CNumber(0.2936, 0.491), new CNumber(0.5877, 0.1148)};
        aIndices = new int[][]{
                {0, 0, 0, 0, 2},
                {0, 0, 1, 0, 2},
                {0, 3, 0, 0, 2},
                {1, 0, 1, 0, 1},
                {1, 0, 1, 0, 4},
                {1, 2, 1, 0, 4},
                {2, 0, 0, 0, 0},
                {2, 0, 1, 0, 2},
                {2, 2, 1, 0, 0},
                {2, 3, 1, 0, 2},
                {2, 3, 1, 0, 3}};
        A = new CooCTensor(aShape, aEntries, aIndices);

        expShape = new Shape(3, 1, 5, 4, 2);
        expEntries = new CNumber[]{new CNumber(0.2252, -0.8303), new CNumber(0.7022, -0.9162), new CNumber(0.6672, -0.2974), new CNumber(0.4421, -0.7193), new CNumber(0.3796, -0.9056), new CNumber(0.2784, -0.3588), new CNumber(0.3264, -0.909), new CNumber(0.7772, -0.396), new CNumber(0.1959, -0.2546), new CNumber(0.2936, -0.491), new CNumber(0.5877, -0.1148)};
        expIndices = new int[][]{
                {0, 0, 2, 0, 0},
                {0, 0, 2, 0, 1},
                {0, 0, 2, 3, 0},
                {1, 0, 1, 0, 1},
                {1, 0, 4, 0, 1},
                {1, 0, 4, 2, 1},
                {2, 0, 0, 0, 0},
                {2, 0, 0, 2, 1},
                {2, 0, 2, 0, 1},
                {2, 0, 2, 3, 1},
                {2, 0, 3, 3, 1}};
        exp = new CooCTensor(expShape, expEntries, expIndices);

        assertEquals(exp, A.H(0, 3, 4, 1, 2));

        // ----------------------- Sub-case 4 -----------------------
        assertThrows(IllegalArgumentException.class, ()->A.H(0, 1, 3, 2));
        assertThrows(IllegalArgumentException.class, ()->A.H(0, 3, 4, 1, 2, 5));
        assertThrows(IllegalArgumentException.class, ()->A.H(0, 3, -4, 1, 2));
        assertThrows(IllegalArgumentException.class, ()->A.H(0, 15, 4, 1, 2));
        assertThrows(IndexOutOfBoundsException.class, ()->A.H(5, 1));
    }


    @Test
    void transposeTests() {
        // ----------------------- Sub-case 1 -----------------------
        aShape = new Shape(3, 4, 2, 1, 5);
        aEntries = new CNumber[]{new CNumber(0.9613, 0.238), new CNumber(0.9947, 0.4532), new CNumber(0.1439, 0.7333), new CNumber(0.2738, 0.7492), new CNumber(0.0159, 0.6496), new CNumber(0.44, 0.5992), new CNumber(0.2544, 0.174), new CNumber(0.0317, 0.3052), new CNumber(0.3788, 0.4169), new CNumber(0.2586, 0.7146), new CNumber(0.148, 0.1819)};
        aIndices = new int[][]{
                {0, 0, 0, 0, 4},
                {0, 0, 1, 0, 1},
                {0, 1, 0, 0, 2},
                {0, 1, 1, 0, 3},
                {1, 1, 1, 0, 0},
                {1, 1, 1, 0, 4},
                {1, 2, 0, 0, 4},
                {1, 2, 1, 0, 1},
                {1, 2, 1, 0, 4},
                {2, 0, 1, 0, 4},
                {2, 1, 1, 0, 4}};
        A = new CooCTensor(aShape, aEntries, aIndices);

        expShape = new Shape(5, 4, 2, 1, 3);
        expEntries = new CNumber[]{new CNumber(0.0159, 0.6496), new CNumber(0.9947, 0.4532), new CNumber(0.0317, 0.3052), new CNumber(0.1439, 0.7333), new CNumber(0.2738, 0.7492), new CNumber(0.9613, 0.238), new CNumber(0.2586, 0.7146), new CNumber(0.44, 0.5992), new CNumber(0.148, 0.1819), new CNumber(0.2544, 0.174), new CNumber(0.3788, 0.4169)};
        expIndices = new int[][]{
                {0, 1, 1, 0, 1},
                {1, 0, 1, 0, 0},
                {1, 2, 1, 0, 1},
                {2, 1, 0, 0, 0},
                {3, 1, 1, 0, 0},
                {4, 0, 0, 0, 0},
                {4, 0, 1, 0, 2},
                {4, 1, 1, 0, 1},
                {4, 1, 1, 0, 2},
                {4, 2, 0, 0, 1},
                {4, 2, 1, 0, 1}};
        exp = new CooCTensor(expShape, expEntries, expIndices);
        assertEquals(exp, A.T());

        // ----------------------- Sub-case 2 -----------------------
        aShape = new Shape(3, 4, 2, 1, 5);
        aEntries = new CNumber[]{new CNumber(0.5902, 0.9131), new CNumber(0.3367, 0.9463), new CNumber(0.4198, 0.1293), new CNumber(0.6835, 0.2796), new CNumber(0.5134, 0.2389), new CNumber(0.7717, 0.3427), new CNumber(0.7304, 0.4389), new CNumber(0.4974, 0.184), new CNumber(0.1768, 0.9627), new CNumber(0.5433, 0.2314), new CNumber(0.9679, 0.4831)};
        aIndices = new int[][]{
                {0, 1, 0, 0, 3},
                {0, 1, 1, 0, 4},
                {0, 2, 0, 0, 1},
                {1, 0, 0, 0, 1},
                {1, 0, 0, 0, 2},
                {1, 2, 0, 0, 3},
                {1, 3, 0, 0, 0},
                {2, 1, 0, 0, 4},
                {2, 1, 1, 0, 0},
                {2, 2, 0, 0, 3},
                {2, 3, 0, 0, 4}};
        A = new CooCTensor(aShape, aEntries, aIndices);

        expShape = new Shape(2, 4, 3, 1, 5);
        expEntries = new CNumber[]{new CNumber(0.6835, 0.2796), new CNumber(0.5134, 0.2389), new CNumber(0.5902, 0.9131), new CNumber(0.4974, 0.184), new CNumber(0.4198, 0.1293), new CNumber(0.7717, 0.3427), new CNumber(0.5433, 0.2314), new CNumber(0.7304, 0.4389), new CNumber(0.9679, 0.4831), new CNumber(0.3367, 0.9463), new CNumber(0.1768, 0.9627)};
        expIndices = new int[][]{
                {0, 0, 1, 0, 1},
                {0, 0, 1, 0, 2},
                {0, 1, 0, 0, 3},
                {0, 1, 2, 0, 4},
                {0, 2, 0, 0, 1},
                {0, 2, 1, 0, 3},
                {0, 2, 2, 0, 3},
                {0, 3, 1, 0, 0},
                {0, 3, 2, 0, 4},
                {1, 1, 0, 0, 4},
                {1, 1, 2, 0, 0}};
        exp = new CooCTensor(expShape, expEntries, expIndices);

        assertEquals(exp, A.T(0, 2));
        assertEquals(exp, A.T(2, 0));

        // ----------------------- Sub-case 3 -----------------------
        aShape = new Shape(3, 4, 2, 1, 5);
        aEntries = new CNumber[]{new CNumber(0.8834, 0.1986), new CNumber(0.3184, 0.3042), new CNumber(0.8551, 0.4776), new CNumber(0.7626, 0.7819), new CNumber(0.1152, 0.4055), new CNumber(0.6564, 0.7552), new CNumber(0.3097, 0.5647), new CNumber(0.3279, 0.7208), new CNumber(0.4838, 0.6065), new CNumber(0.8963, 0.7191), new CNumber(0.2443, 0.4567)};
        aIndices = new int[][]{
                {0, 0, 0, 0, 3},
                {0, 0, 1, 0, 0},
                {0, 1, 0, 0, 4},
                {0, 2, 0, 0, 2},
                {1, 0, 0, 0, 3},
                {1, 2, 0, 0, 0},
                {1, 2, 1, 0, 3},
                {2, 1, 1, 0, 4},
                {2, 2, 0, 0, 4},
                {2, 2, 1, 0, 0},
                {2, 3, 1, 0, 1}};
        A = new CooCTensor(aShape, aEntries, aIndices);

        expShape = new Shape(3, 1, 5, 4, 2);
        expEntries = new CNumber[]{new CNumber(0.3184, 0.3042), new CNumber(0.7626, 0.7819), new CNumber(0.8834, 0.1986), new CNumber(0.8551, 0.4776), new CNumber(0.6564, 0.7552), new CNumber(0.1152, 0.4055), new CNumber(0.3097, 0.5647), new CNumber(0.8963, 0.7191), new CNumber(0.2443, 0.4567), new CNumber(0.3279, 0.7208), new CNumber(0.4838, 0.6065)};
        expIndices = new int[][]{
                {0, 0, 0, 0, 1},
                {0, 0, 2, 2, 0},
                {0, 0, 3, 0, 0},
                {0, 0, 4, 1, 0},
                {1, 0, 0, 2, 0},
                {1, 0, 3, 0, 0},
                {1, 0, 3, 2, 1},
                {2, 0, 0, 2, 1},
                {2, 0, 1, 3, 1},
                {2, 0, 4, 1, 1},
                {2, 0, 4, 2, 0}};
        exp = new CooCTensor(expShape, expEntries, expIndices);

        assertEquals(exp, A.T(0, 3, 4, 1, 2));

        // ----------------------- Sub-case 4 -----------------------
        assertThrows(IllegalArgumentException.class, ()->A.T(0, 1, 3, 2));
        assertThrows(IllegalArgumentException.class, ()->A.T(0, 3, 4, 1, 2, 5));
        assertThrows(IllegalArgumentException.class, ()->A.T(0, 3, -4, 1, 2));
        assertThrows(IllegalArgumentException.class, ()->A.T(0, 15, 4, 1, 2));
        assertThrows(IndexOutOfBoundsException.class, ()->A.T(5, 1));
    }
}
