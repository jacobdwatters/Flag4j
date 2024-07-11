package org.flag4j.vector;

import org.flag4j.complex_numbers.CNumber;
import org.flag4j.dense.CVector;
import org.flag4j.dense.Vector;
import org.flag4j.sparse.CooCVector;
import org.flag4j.sparse.CooVector;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class VectorElemMultDivTests {

    int[] indices;

    double[] aEntries;
    Vector a;

    @Test
    void realDenseMultTestCase() {
        double[] bEntries, expEntries;
        Vector b, exp;

        // ------------------------ Sub-case 1 ------------------------
        aEntries = new double[]{1.234, -9.4, 8.45};
        a = new Vector(aEntries);
        bEntries = new double[]{8.55, -9.133, -8.34};
        b = new Vector(bEntries);
        expEntries = new double[]{aEntries[0]*bEntries[0], aEntries[1]*bEntries[1], aEntries[2]*bEntries[2]};
        exp = new Vector(expEntries);

        assertEquals(exp, a.elemMult(b));

        // ------------------------ Sub-case 2 ------------------------
        aEntries = new double[]{1.234, -9.4, 8.45};
        a = new Vector(aEntries);
        bEntries = new double[]{8.55, -9.133, -8.34, 23.4};
        b = new Vector(bEntries);

        Vector finalB = b;
        assertThrows(LinearAlgebraException.class, ()->a.elemMult(finalB));
    }


    @Test
    void complexDenseMultTestCase() {
        CNumber[] bEntries, expEntries;
        CVector b, exp;

        // ------------------------ Sub-case 1 ------------------------
        aEntries = new double[]{1.234, -9.4, 8.45};
        a = new Vector(aEntries);
        bEntries = new CNumber[]{new CNumber(23.456, -9234), new CNumber(0, 8.234), new CNumber(-9234.5, 0.24)};
        b = new CVector(bEntries);
        expEntries = new CNumber[]{bEntries[0].mult(aEntries[0]),bEntries[1].mult(aEntries[1]), bEntries[2].mult(aEntries[2])};
        exp = new CVector(expEntries);

        assertEquals(exp, a.elemMult(b));

        // ------------------------ Sub-case 2 ------------------------
        aEntries = new double[]{1.234, -9.4, 8.45};
        a = new Vector(aEntries);
        bEntries = new CNumber[]{new CNumber(23.456, -9234), new CNumber(0, 8.234)};
        b = new CVector(bEntries);

        CVector finalB = b;
        assertThrows(LinearAlgebraException.class, ()->a.elemMult(finalB));
    }


    @Test
    void realSparseMultTestCase() {
        int[] expIndices;
        double[] bEntries, expEntries;
        CooVector b, exp;

        // ------------------------ Sub-case 1 ------------------------
        aEntries = new double[]{1.234, -9.4, 8.45};
        a = new Vector(aEntries);
        bEntries = new double[]{8.55};
        indices = new int[]{2};
        b = new CooVector(3, bEntries, indices);
        expEntries = new double[]{aEntries[2]*bEntries[0]};
        expIndices = new int[]{2};
        exp = new CooVector(3, expEntries, expIndices);

        CooVector act = a.elemMult(b);

        assertEquals(exp.size, act.size);
        assertArrayEquals(exp.entries, act.entries);
        assertArrayEquals(exp.indices, act.indices);

        // ------------------------ Sub-case 2 ------------------------
        aEntries = new double[]{1.234, -9.4, 8.45};
        a = new Vector(aEntries);
        bEntries = new double[]{8.55};
        indices = new int[]{2};
        b = new CooVector(402, bEntries, indices);

        CooVector finalB = b;
        assertThrows(LinearAlgebraException.class, ()->a.elemMult(finalB));
    }


    @Test
    void complexSparseMultTestCase() {
        int[] expIndices;
        CNumber[] bEntries, expEntries;
        CooCVector b, exp;

        // ------------------------ Sub-case 1 ------------------------
        aEntries = new double[]{1.234, -9.4, 8.45};
        a = new Vector(aEntries);
        bEntries = new CNumber[]{new CNumber(-9.234, 1.5)};
        indices = new int[]{2};
        b = new CooCVector(3, bEntries, indices);
        expEntries = new CNumber[]{bEntries[0].mult(aEntries[2])};
        expIndices = new int[]{2};
        exp = new CooCVector(3, expEntries, expIndices);

        CooCVector act = a.elemMult(b);

        assertEquals(exp.size, act.size);
        assertArrayEquals(exp.entries, act.entries);
        assertArrayEquals(exp.indices, act.indices);

        // ------------------------ Sub-case 2 ------------------------
        aEntries = new double[]{1.234, -9.4, 8.45};
        a = new Vector(aEntries);
        bEntries = new CNumber[]{new CNumber(-9.234, 1.5)};
        indices = new int[]{2};
        b = new CooCVector(31, bEntries, indices);

        CooCVector finalB = b;
        assertThrows(LinearAlgebraException.class, ()->a.elemMult(finalB));
    }

    // ---------------------------------------------------------------------------------------------------------------
    @Test
    void realDenseDivTestCase() {
        double[] bEntries, expEntries;
        Vector b, exp;

        // ------------------------ Sub-case 1 ------------------------
        aEntries = new double[]{1.234, -9.4, 8.45};
        a = new Vector(aEntries);
        bEntries = new double[]{8.55, -9.133, -8.34};
        b = new Vector(bEntries);
        expEntries = new double[]{aEntries[0]/bEntries[0], aEntries[1]/bEntries[1], aEntries[2]/bEntries[2]};
        exp = new Vector(expEntries);

        assertEquals(exp, a.elemDiv(b));

        // ------------------------ Sub-case 2 ------------------------
        aEntries = new double[]{1.234, -9.4, 8.45};
        a = new Vector(aEntries);
        bEntries = new double[]{8.55, -9.133, -8.34, 23.4};
        b = new Vector(bEntries);

        Vector finalB = b;
        assertThrows(LinearAlgebraException.class, ()->a.elemDiv(finalB));
    }


    @Test
    void complexDenseDivTestCase() {
        CNumber[] bEntries, expEntries;
        CVector b, exp;

        // ------------------------ Sub-case 1 ------------------------
        aEntries = new double[]{1.234, -9.4, 8.45};
        a = new Vector(aEntries);
        bEntries = new CNumber[]{new CNumber(23.456, -9234), new CNumber(0, 8.234), new CNumber(-9234.5, 0.24)};
        b = new CVector(bEntries);
        expEntries = new CNumber[]{new CNumber(aEntries[0]).div(bEntries[0]),
                new CNumber(aEntries[1]).div(bEntries[1]), new CNumber(aEntries[2]).div(bEntries[2])};
        exp = new CVector(expEntries);

        assertEquals(exp, a.elemDiv(b));

        // ------------------------ Sub-case 2 ------------------------
        aEntries = new double[]{1.234, -9.4, 8.45};
        a = new Vector(aEntries);
        bEntries = new CNumber[]{new CNumber(23.456, -9234), new CNumber(0, 8.234)};
        b = new CVector(bEntries);

        CVector finalB = b;
        assertThrows(LinearAlgebraException.class, ()->a.elemDiv(finalB));
    }
}
