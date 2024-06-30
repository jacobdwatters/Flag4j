package org.flag4j.sparse_complex_vector;

import org.flag4j.complex_numbers.CNumber;
import org.flag4j.core.Shape;
import org.flag4j.sparse.CooCVector;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CooCVectorReshapeTests {
    static int[] aIndices, expIndices;
    static CNumber[] aEntries, expEntries;
    static int sparseSize;
    static CooCVector a, exp;


    @BeforeAll
    static void setup() {
        aEntries = new CNumber[]{
                new CNumber(2.455, -83.6), new CNumber(0, 24.56),
                new CNumber(24.56), new CNumber(-9356.1, 35)
        };
        aIndices = new int[]{4, 56, 9903, 14643};
        sparseSize = 24_023;
        a = new CooCVector(sparseSize, aEntries, aIndices);
    }


    @Test
    void reshapeTestCase() {
        // ------------------ Sub-case 1 ------------------
        expEntries = new CNumber[]{
                new CNumber(2.455, -83.6), new CNumber(0, 24.56),
                new CNumber(24.56), new CNumber(-9356.1, 35)
        };
        expIndices = new int[]{4, 56, 9903, 14643};
        exp = new CooCVector(sparseSize, expEntries, expIndices);

        assertEquals(exp, a.reshape(new Shape(sparseSize)));
        assertEquals(exp, a.reshape(sparseSize));
        assertEquals(exp, a.flatten());
        assertEquals(exp, a.flatten(0));

        // ------------------ Sub-case 2 ------------------
        assertThrows(IllegalArgumentException.class, ()->a.reshape(new Shape(sparseSize-2)));
        assertThrows(IllegalArgumentException.class, ()->a.reshape(sparseSize+32));
        assertThrows(IllegalArgumentException.class, ()->a.flatten(-35));
        assertThrows(IllegalArgumentException.class, ()->a.flatten(1));
    }
}
