package org.flag4j.tensor;


import org.flag4j.complex_numbers.CNumber;
import org.flag4j.core.Shape;
import org.flag4j.dense.CTensor;
import org.flag4j.dense.Tensor;
import org.flag4j.sparse.CooCTensor;
import org.flag4j.sparse.CooTensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;

class TensorEqualsTests {

    static double[] aEntries;
    static Tensor A;
    static Shape aShape, bShape;

    int[][] sparseIndices;

    static void denseSetup() {
        aEntries = new double[]{
                1.23, 2.556, -121.5, 15.61, 14.15, -99.23425,
                0.001345, 2.677, 8.14, -0.000194, 1, 234
        };
        aShape = new Shape(2, 3, 2);
        A = new Tensor(aShape, aEntries);
    }


    static void sparseSetup() {
        aEntries = new double[]{
                1.23, 0, 0, 0, 0, -99.23425,
                0, 2.677, 0, -0.000194, 0, 0
        };
        aShape = new Shape(2, 3, 2);
        A = new Tensor(aShape, aEntries);
    }


    @Test
    void realDenseEqualsTestCase() {
        denseSetup();

        double[] bEntries;
        Tensor B;

        // ---------------------- Sub-case 1 ----------------------
        bEntries = new double[]{
                -0.00234, 15.6, 99.2442, 100.252, -78.2556, 0.111134,
                671.455, -0.00024, 515.667, 14.515, 100.135, 0
        };
        bShape = new Shape(2, 3, 2);
        B = new Tensor(bShape, bEntries);

        assertNotEquals(A, B);

        // ---------------------- Sub-case 2 ----------------------
        bEntries = new double[]{
                1.23, 2.556, -121.5, 15.61, 14.15, -99.23425,
                0.001345, 2.677, 8.14, -0.000194, 1, 234
        };
        bShape = new Shape(2, 3, 2);
        B = new Tensor(bShape, bEntries);

        assertEquals(A, B);

        // ---------------------- Sub-case 3 ----------------------
        bEntries = new double[]{
                1.23, 2.556, -121.5, 15.61, 14.15, -99.23425,
                0.001345, 2.677, 8.14, -0.000194, 1, 234
        };
        bShape = new Shape(4, 3);
        B = new Tensor(bShape, bEntries);

        assertNotEquals(A, B);

        // ---------------------- Sub-case 4 ----------------------
        bEntries = new double[]{
                1.23, 2.556, -121.5, 15.61, 14.15, -99.23425,
                0.001345, 2.677, 8.14, -0.000194, 1, 234
        };
        bShape = new Shape(2, 1, 3, 2);
        B = new Tensor(bShape, bEntries);

        assertNotEquals(A, B);
    }


    @Test
    void sparseDenseEqualsTestCase() {
        sparseSetup();

        double[] bEntries;
        CooTensor B;

        // ---------------------- Sub-case 1 ----------------------
        bEntries = new double[]{1.23, -99.23425, 2.677, -0.000194};
        bShape = new Shape(2, 3, 2);
        sparseIndices = new int[][]{
                aShape.getIndices(0),
                aShape.getIndices(5),
                aShape.getIndices(7),
                aShape.getIndices(9)
        };
        B = new CooTensor(bShape, bEntries, sparseIndices);

        assertEquals(A, B);

        // ---------------------- Sub-case 2 ----------------------
        bEntries = new double[]{1.23, -99.25, 2.677, -0.000194};
        bShape = new Shape(2, 3, 2);
        sparseIndices = new int[][]{
                aShape.getIndices(0),
                aShape.getIndices(5),
                aShape.getIndices(7),
                aShape.getIndices(9)
        };
        B = new CooTensor(bShape, bEntries, sparseIndices);

        assertNotEquals(A, B);

        // ---------------------- Sub-case 3 ----------------------
        bEntries = new double[]{1.23, -99.23425, 2.677, -0.000194};
        bShape = new Shape(21, 31, 2, 10005);
        sparseIndices = new int[][]{
                bShape.getIndices(0),
                bShape.getIndices(5),
                bShape.getIndices(7),
                bShape.getIndices(9)
        };
        B = new CooTensor(bShape, bEntries, sparseIndices);

        assertNotEquals(A, B);

        // ---------------------- Sub-case 4 ----------------------
        bEntries = new double[]{1.23, -99.25, 2.677, -0.000194};
        bShape = new Shape(21, 3, 24);
        sparseIndices = new int[][]{
                aShape.getIndices(0),
                aShape.getIndices(5),
                aShape.getIndices(7),
                aShape.getIndices(9)
        };
        B = new CooTensor(bShape, bEntries, sparseIndices);

        assertNotEquals(A, B);
    }


    @Test
    void complexDenseEqualsTestCase() {
        denseSetup();

        CNumber[] bEntries;
        CTensor B;

        // ----------------------- Sub-case 1 -----------------------
        bEntries = new CNumber[]{
                new CNumber(1.23), new CNumber(2.556), new CNumber(-121.5),
                new CNumber(15.61), new CNumber(14.15), new CNumber(-99.23425),
                new CNumber(0.001345), new CNumber(2.677), new CNumber(8.14),
                new CNumber(-0.000194), new CNumber(1), new CNumber(234)
        };
        bShape = new Shape(2, 3, 2);
        B = new CTensor(bShape, bEntries);

        assertEquals(A, B);

        // ----------------------- Sub-case 2 -----------------------
        bEntries = new CNumber[]{
                new CNumber(1.23), new CNumber(2.556), new CNumber(-121.5),
                new CNumber(15.61, 1.4), new CNumber(14.15), new CNumber(-99.23425),
                new CNumber(0.001345), new CNumber(2.677), new CNumber(8.14),
                new CNumber(-0.000194), new CNumber(1), new CNumber(234)
        };
        bShape = new Shape(2, 3, 2);
        B = new CTensor(bShape, bEntries);

        assertNotEquals(A, B);

        // ----------------------- Sub-case 3 -----------------------
        bEntries = new CNumber[]{
                new CNumber(1.23), new CNumber(2.556), new CNumber(-121.5),
                new CNumber(15.61), new CNumber(14.15), new CNumber(-99.23425),
                new CNumber(0.001345), new CNumber(2.677), new CNumber(8.14),
                new CNumber(-0.000194), new CNumber(1), new CNumber(234)
        };
        bShape = new Shape(4, 3);
        B = new CTensor(bShape, bEntries);

        assertNotEquals(A, B);

        // ----------------------- Sub-case 4 -----------------------
        bEntries = new CNumber[]{
                new CNumber(1.23), new CNumber(2.556), new CNumber(-121.5),
                new CNumber(15.61), new CNumber(14.15), new CNumber(-99.23425),
                new CNumber(9.245), new CNumber(2.677), new CNumber(8.14),
                new CNumber(-0.000194), new CNumber(1), new CNumber(234)
        };
        bShape = new Shape(2, 3, 2);
        B = new CTensor(bShape, bEntries);

        assertNotEquals(A, B);
    }

    
    @Test
    void complexSparseEqualsTestCase() {
        sparseSetup();

        CNumber[] bEntries;
        CooCTensor B;

        // ---------------------- Sub-case 1 ----------------------
        bEntries = new CNumber[]{
                new CNumber(1.23), new CNumber(-99.23425),
                new CNumber(2.677), new CNumber(-0.000194)};
        bShape = new Shape(2, 3, 2);
        sparseIndices = new int[][]{
                aShape.getIndices(0),
                aShape.getIndices(5),
                aShape.getIndices(7),
                aShape.getIndices(9)
        };
        B = new CooCTensor(bShape, bEntries, sparseIndices);

        assertEquals(A, B);

        // ---------------------- Sub-case 2 ----------------------
        bEntries = new CNumber[]{
                new CNumber(1.23), new CNumber(-99.23425, 1.34235),
                new CNumber(2.677), new CNumber(-0.000194)};
        bShape = new Shape(2, 3, 2);
        sparseIndices = new int[][]{
                aShape.getIndices(0),
                aShape.getIndices(5),
                aShape.getIndices(7),
                aShape.getIndices(9)
        };
        B = new CooCTensor(bShape, bEntries, sparseIndices);

        assertNotEquals(A, B);

        // ---------------------- Sub-case 3 ----------------------
        bEntries = new CNumber[]{
                new CNumber(1.23), new CNumber(-99.23425),
                new CNumber(2.677), new CNumber(-0.000194)};
        bShape = new Shape(21, 31, 2, 10005);
        sparseIndices = new int[][]{
                bShape.getIndices(0),
                bShape.getIndices(5),
                bShape.getIndices(7),
                bShape.getIndices(9)
        };
        B = new CooCTensor(bShape, bEntries, sparseIndices);

        assertNotEquals(A, B);

        // ---------------------- Sub-case 4 ----------------------
        bEntries = new CNumber[]{
                new CNumber(1.23), new CNumber(-99.23425),
                new CNumber(2.677), new CNumber(-0.000194)};
        bShape = new Shape(21, 3, 24);
        sparseIndices = new int[][]{
                aShape.getIndices(0),
                aShape.getIndices(5),
                aShape.getIndices(7),
                aShape.getIndices(9)
        };
        B = new CooCTensor(bShape, bEntries, sparseIndices);

        assertNotEquals(A, B);
    }


    @Test
    void objectTestCase() {
        denseSetup();

        // ---------------------- Sub-case 1 ----------------------
        assertNotEquals(A, new Shape(1, 14));

        // ---------------------- Sub-case 2 ----------------------
        assertNotEquals(A, Double.valueOf(2.245));

        // ---------------------- Sub-case 3 ----------------------
        assertNotEquals(A, "Hello Flag4j");
    }
}
