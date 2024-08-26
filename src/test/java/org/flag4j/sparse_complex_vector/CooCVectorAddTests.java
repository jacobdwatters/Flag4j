package org.flag4j.sparse_complex_vector;

import org.flag4j.arrays_old.dense.CVectorOld;
import org.flag4j.arrays_old.dense.VectorOld;
import org.flag4j.arrays_old.sparse.CooCVectorOld;
import org.flag4j.arrays_old.sparse.CooVectorOld;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CooCVectorAddTests {
    CooCVectorOld a;

    @Test
    void sparseAddTestCase() {
        CooVectorOld b;
        CooCVectorOld exp;

        CNumber[] aValues = {new CNumber(32.5, 98), new CNumber(-8.2, 55.1), new CNumber(0, 14.5)};
        int[] aIndices = {0, 5, 103};
        int size = 304;
        a = new CooCVectorOld(size, aValues, aIndices);

        double[] bValues = {44, -5.66, 22.445, -0.994, 10.5};
        int[] bIndices = {1, 5, 11, 67, 200};
        b = new CooVectorOld(size, bValues, bIndices);

        // --------------------- Sub-case 1 ---------------------
        CNumber[] expValues = {new CNumber(32.5, 98), new CNumber(44), new CNumber(-8.2, 55.1).add(-5.66),
                new CNumber(22.445), new CNumber(-0.994), new CNumber(0, 14.5), new CNumber(10.50)};
        int[] expIndices = {0, 1, 5, 11, 67, 103, 200};
        exp = new CooCVectorOld(size, expValues, expIndices);

        assertEquals(exp, a.add(b));

        // --------------------- Sub-case 2 ---------------------
        bValues = new double[]{44, -5.66, 22.445, -0.994, 10.5};
        bIndices = new int[]{1, 5, 11, 67, 200};
        b = new CooVectorOld(size+13, bValues, bIndices);

        CooVectorOld finalB = b;
        assertThrows(LinearAlgebraException.class, ()->a.add(finalB));
    }


    @Test
    void sparseComplexAddTestCase() {
        CooCVectorOld b, exp;

        CNumber[] aValues = {new CNumber(32.5, 98), new CNumber(-8.2, 55.1), new CNumber(0, 14.5)};
        int[] aIndices = {0, 5, 103};
        int size = 304;
        a = new CooCVectorOld(size, aValues, aIndices);

        CNumber[] bValues = {new CNumber(1, -0.024),
                new CNumber(99.24, 1.5), new CNumber(0, 1.4)};
        int[] bIndices = {1, 5, 6};
        b = new CooCVectorOld(size, bValues, bIndices);

        // --------------------- Sub-case 1 ---------------------
        CNumber[] expValues = {new CNumber(32.5, 98), new CNumber(1, -0.024),
                new CNumber(-8.2, 55.1).add(new CNumber(99.24, 1.5)), new CNumber(0, 1.4),
                new CNumber(0, 14.5)
        };
        int[] expIndices = {0, 1, 5, 6, 103};
        exp = new CooCVectorOld(size, expValues, expIndices);

        assertEquals(exp, a.add(b));

        // --------------------- Sub-case 2 ---------------------
        bValues = new CNumber[]{new CNumber(1, -0.024),
                new CNumber(99.24, 1.5), new CNumber(0, 1.4)};
        bIndices = new int[]{1, 5, 6};
        b = new CooCVectorOld(size+13, bValues, bIndices);

        CooCVectorOld finalB = b;
        assertThrows(LinearAlgebraException.class, ()->a.add(finalB));
    }


    @Test
    void denseTestCase() {
        VectorOld b;
        CVectorOld exp;

        CNumber[] aValues = {new CNumber(32.5, 98), new CNumber(-8.2, 55.1), new CNumber(0, 14.5)};
        int[] aIndices = {0, 2, 5};
        int size = 8;
        a = new CooCVectorOld(size, aValues, aIndices);

        double[] bValues = {1, 5, -0.0024, 1, 2001.256, 61, -99.24, 1.5};
        b = new VectorOld(bValues);

        // --------------------- Sub-case 1 ---------------------
        CNumber[] expValues = {
                new CNumber(1).add(aValues[0]), new CNumber(5),
                new CNumber(-0.0024).add(aValues[1]), new CNumber(1),
                new CNumber(2001.256), new CNumber(61).add(aValues[2]),
                new CNumber(-99.24), new CNumber(1.5)};
        exp = new CVectorOld(expValues);

        assertEquals(exp, a.add(b));

        // --------------------- Sub-case 2 ---------------------
        bValues = new double[]{1, 5, -0.0024, 1, 2001.256, 61};
        b = new VectorOld(bValues);

        VectorOld finalB = b;
        assertThrows(LinearAlgebraException.class, ()->a.add(finalB));
    }


    @Test
    void denseComplexTestCase() {
        CVectorOld b, exp;

        CNumber[] aValues = {new CNumber(32.5, 98), new CNumber(-8.2, 55.1)};
        int[] aIndices = {0, 2};
        int size = 5;
        a = new CooCVectorOld(size, aValues, aIndices);

        CNumber[] bValues = {new CNumber(1.445, -9.24), new CNumber(1.45),
                new CNumber(0, -99.145), new CNumber(4.51, 8.456), new CNumber(11.34, -0.00245)};
        b = new CVectorOld(bValues);

        // --------------------- Sub-case 1 ---------------------
        CNumber[] expValues = {new CNumber(1.445, -9.24).add(new CNumber(32.5, 98)), new CNumber(1.45),
                new CNumber(0, -99.145).add(new CNumber(-8.2, 55.1)), new CNumber(4.51, 8.456), new CNumber(11.34, -0.00245)};
        exp = new CVectorOld(expValues);

        assertEquals(exp, a.add(b));

        // --------------------- Sub-case 2 ---------------------
        bValues = new CNumber[]{new CNumber(1.445, -9.24), new CNumber(1.45),
                new CNumber(0, -99.145), new CNumber(4.51, 8.456),
                new CNumber(11.34, -0.00245), new CNumber(34.5, 0.0014)};
        b = new CVectorOld(bValues);

        CVectorOld finalB = b;
        assertThrows(LinearAlgebraException.class, ()->a.add(finalB));
    }


    @Test
    void scalarTestCase() {
        double b;
        CVectorOld exp;

        CNumber[] aValues = {new CNumber(32.5, 98), new CNumber(-8.2, 55.1), new CNumber(0, 14.5)};
        int[] aIndices = {0, 2, 5};
        int size = 8;
        a = new CooCVectorOld(size, aValues, aIndices);

        b = 2.345;

        // --------------------- Sub-case 1 ---------------------
        CNumber[] expValues = {aValues[0].add(new CNumber(b)), new CNumber(b), aValues[1].add(new CNumber(b)), new CNumber(b),
                new CNumber(b), aValues[2].add(new CNumber(b)), new CNumber(b), new CNumber(b)};
        exp = new CVectorOld(expValues);

        assertEquals(exp, a.add(b));
    }


    @Test
    void complexScalarTestCase() {
        CNumber b;
        CVectorOld exp;

        CNumber[] aValues = {new CNumber(32.5, 98), new CNumber(-8.2, 55.1), new CNumber(0, 14.5)};
        int[] aIndices = {0, 2, 3};
        int size = 5;
        a = new CooCVectorOld(size, aValues, aIndices);

        b = new CNumber(13.455, -1459.4521);

        // --------------------- Sub-case 1 ---------------------
        CNumber[] expValues = {new CNumber(32.5, 98).add(b), b, new CNumber(-8.2, 55.1).add(b),
                new CNumber(0, 14.5).add(b), b};
        exp = new CVectorOld(expValues);

        assertEquals(exp, a.add(b));
    }
}
