/*
*	@file
*	@author  Thomas Kroes <t.kroes at tudelft.nl>
*	@version 1.0
*	
*	@section LICENSE
*	
*	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
*	
*	Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
*	Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
*	Neither the name of the TU Delft nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
*
*	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

namespace ExposureRender
{

/*! Adds a function to a class that returns the value of member \a name of \a type */
#define GET_MACRO(scope,name,type)	 										\
scope type Get##name() const												\
{																			\
	return this->name;														\
}

/*! Adds a function to a class that returns a reference to member a\ name of \a type */
#define GET_REF_MACRO(scope,name,type)										\
scope type& Get##name()														\
{																			\
	return this->name;														\
}

/*! Adds a function to a class that returns a reference to member a\ name of \a type */
#define GET_PTR_MACRO(scope,name,type)										\
scope type* Get##name()														\
{																			\
	return &(this->name);													\
}

/*! Adds a function to a class that sets member a\ name of \a type */
#define SET_MACRO(scope,name,type)											\
scope void Set##name(const type& Arg)										\
{																			\
	this->name = Arg;														\
}

/*! Adds a function to a class that sets member a\ name of \a type, and flag the time stamp as modified */
#define SET_TS_MACRO(scope,name,type)										\
scope void Set##name(const type& Arg)										\
{																			\
	this->name = Arg;														\
}

/*! Adds a function to a class that returns the value of an element of a 2D array \a name of \a type */
#define GET_2D_ARRAY_ELEMENT_MACRO(scope,name,type)							\
scope type Get##name(const int& x, const int& y) const						\
{																			\
	return this->name[x][y];												\
}

/*! Adds a function to a class that returns the reference to an element of a 2D array \a name of \a type */
#define GET_2D_ARRAY_ELEMENT_REF_MACRO(scope,name,type)						\
scope type& Get##name(const int& x, const int& y)							\
{																			\
	return this->name[x][y];												\
}

/*! Adds a function to a class that sets the value of an element of a 2D array \a name of \a type */
#define SET_2D_ARRAY_ELEMENT_MACRO(scope,name,type)							\
scope type Set##name(const int& x, const int& y, const type& Arg)			\
{																			\
	this->name[x][y] = Arg;													\
}

/*! Adds a function to a class for getting member a\ name and setting it */
#define GET_SET_MACRO(scope,name,type)										\
scope GET_MACRO(scope,name,type)											\
scope SET_MACRO(scope,name,type)

/*! Adds a function to a class for getting member a\ name and setting it */
#define GET_SET_TS_MACRO(scope,name,type)									\
scope GET_MACRO(scope,name,type)											\
scope SET_TS_MACRO(scope,name,type)

/*! Adds a function to a class for getting member a\ name by reference and setting it */
#define GET_REF_SET_MACRO(scope,name,type)									\
scope GET_REF_MACRO(scope,name,type)										\
scope SET_MACRO(scope,name,type)

/*! Adds a function to a class for getting member a\ name by reference and setting it */
#define GET_REF_SET_TS_MACRO(scope,name,type)								\
scope GET_REF_MACRO(scope,name,type)										\
scope SET_TS_MACRO(scope,name,type)

}
