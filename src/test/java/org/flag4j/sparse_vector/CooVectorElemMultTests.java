package org.flag4j.sparse_vector;

import org.flag4j.arrays_old.dense.CVectorOld;
import org.flag4j.arrays_old.dense.VectorOld;
import org.flag4j.arrays_old.sparse.CooCVectorOld;
import org.flag4j.arrays_old.sparse.CooVectorOld;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CooVectorElemMultTests {

    CooVectorOld a;
    int size;

    @Test
    void sparseElemMultTestCase() {
        double[] aValues = {1.34, 51.6, -0.00245, 99.2456, -1005.6};
        int[] aIndices = {2, 5, 81, 102, 104};
        size = 151;
        a = new CooVectorOld(size, aValues, aIndices);

        double[] bValues, expValues;
        int[] bIndices, expIndices;
        CooVectorOld b, exp;

        // -------------------- Sub-case 1 --------------------
        bValues = new double[]{1.223, 44.51, 3.4};
        bIndices = new int[]{2, 81, 103};
        b = new CooVectorOld(size, bValues, bIndices);

        expValues = new double[]{1.223*1.34, 44.51*-0.00245};
        expIndices = new int[]{2, 81};
        exp = new CooVectorOld(151, expValues, expIndices);
        assertEquals(exp, a.elemMult(b));

        // -------------------- Sub-case 2 --------------------
        bValues = new double[]{1.223, 44.51, 3.4};
        bIndices = new int[]{2, 81, 103};
        b = new CooVectorOld(size-23, bValues, bIndices);

        CooVectorOld finalB = b;
        assertThrows(LinearAlgebraException.class, ()->a.elemMult(finalB));
    }


    @Test
    void denseElemMultTestCase() {
        double[] aValues = {1.34, 51.6, -0.00245};
        int[] aIndices = {0, 2, 3};
        size = 7;
        a = new CooVectorOld(size, aValues, aIndices);

        double[] bValues, expValues;
        int[] expIndices;
        VectorOld b;
        CooVectorOld exp;

        // -------------------- Sub-case 1 --------------------
        bValues = new double[]{1.223, -44.51, 3.4, 2.3, 14.5, -14.51, 0.14};
        b = new VectorOld(bValues);

        expValues = new double[]{1.34*1.223, 51.6*3.4, -0.00245*2.3};
        expIndices = new int[]{0, 2, 3};
        exp = new CooVectorOld(size, expValues, expIndices);
        assertEquals(exp, a.elemMult(b));

        // -------------------- Sub-case 2 --------------------
        bValues = new double[]{1.223, -44.51, 3.4, 2.3, 14.5, -14.51};
        b = new VectorOld(bValues);

        VectorOld finalB = b;
        assertThrows(LinearAlgebraException.class, ()->a.elemMult(finalB));
    }


    @Test
    void sparseComplexElemMultTestCase() {
        double[] aValues = {1.34, 51.6, -0.00245, 99.2456, -1005.6};
        int[] aIndices = {2, 5, 81, 102, 104};
        size = 151;
        a = new CooVectorOld(size, aValues, aIndices);

        CNumber[] bValues, expValues;
        int[] bIndices, expIndices;
        CooCVectorOld b, exp;

        // -------------------- Sub-case 1 --------------------
        bValues = new CNumber[]{new CNumber(2.441, -9.245), new CNumber(0, 4.51), new CNumber(24.5)};
        bIndices = new int[]{2, 81, 103};
        b = new CooCVectorOld(size, bValues, bIndices);

        expValues = new CNumber[]{
                new CNumber(2.441, -9.245).mult(1.34),
                new CNumber(0, 4.51).mult(-0.00245)};
        expIndices = new int[]{2, 81};
        exp = new CooCVectorOld(151, expValues, expIndices);
        assertEquals(exp, a.elemMult(b));

        // -------------------- Sub-case 2 --------------------
        bValues = new CNumber[]{new CNumber(2.441, -9.245), new CNumber(0, 4.51), new CNumber(24.5)};
        bIndices = new int[]{2, 81, 103};
        b = new CooCVectorOld(size+134, bValues, bIndices);

        CooCVectorOld finalB = b;
        assertThrows(LinearAlgebraException.class, ()->a.elemMult(finalB));
    }


    @Test
    void denseComplexElemMultTestCase() {
        double[] aValues = {1.34, 51.6, -0.00245};
        int[] aIndices = {0, 2, 3};
        size = 5;
        a = new CooVectorOld(size, aValues, aIndices);

        CNumber[] bValues, expValues;
        int[] expIndices;
        CVectorOld b;
        CooCVectorOld exp;

        // -------------------- Sub-case 1 --------------------
        bValues = new CNumber[]{new CNumber(24.3, -0.013), new CNumber(0, 13.6),
                new CNumber(2.4), new CNumber(-994.1 ,1.45), new CNumber(1495, 13.4)};
        b = new CVectorOld(bValues);

        expValues = new CNumber[]{new CNumber(24.3, -0.013).mult(1.34),
                new CNumber(2.4).mult(51.6), new CNumber(-994.1 ,1.45).mult(-0.00245)};
        expIndices = new int[]{0, 2, 3};
        exp = new CooCVectorOld(size, expValues, expIndices);
        assertEquals(exp, a.elemMult(b));

        // -------------------- Sub-case 2 --------------------
        bValues = new CNumber[]{new CNumber(24.3, -0.013), new CNumber(0, 13.6), new CNumber(24),
                new CNumber(2.4), new CNumber(-994.1 ,1.45), new CNumber(1495, 13.4)};
        b = new CVectorOld(bValues);

        CVectorOld finalB = b;
        assertThrows(LinearAlgebraException.class, ()->a.elemMult(finalB));
    }


    @Test
    void doubleScalarElemMultTestCase() {
        double[] aValues = {1.34, 51.6, -0.00245, 99.2456, -1005.6};
        int[] aIndices = {2, 5, 81, 102, 104};
        size = 151;
        a = new CooVectorOld(size, aValues, aIndices);

        double b;

        double[] expValues;
        int[] expIndices;
        CooVectorOld exp;

        // -------------------- Sub-case 1 --------------------
        b = 24.56;

        expValues = new double[]{1.34*b, 51.6*b, -0.00245*b, 99.2456*b, -1005.6*b};
        expIndices = new int[]{2, 5, 81, 102, 104};
        exp = new CooVectorOld(151, expValues, expIndices);
        assertEquals(exp, a.mult(b));
    }


    @Test
    void complexScalarElemMultTestCase() {
        double[] aValues = {1.34, 51.6, -0.00245, 99.2456, -1005.6};
        int[] aIndices = {2, 5, 81, 102, 104};
        size = 151;
        a = new CooVectorOld(size, aValues, aIndices);

        CNumber b;

        CNumber[] expValues;
        int[] expIndices;
        CooCVectorOld exp;

        // -------------------- Sub-case 1 --------------------
        b = new CNumber(234.6677, -9.35);

        expValues = new CNumber[]{b.mult(1.34), b.mult(51.6), b.mult(-0.00245), b.mult(99.2456), b.mult(-1005.6)};
        expIndices = new int[]{2, 5, 81, 102, 104};
        exp = new CooCVectorOld(151, expValues, expIndices);
        assertEquals(exp, a.mult(b));
    }
}
