/* This file is part of the Zenipex Library (zenilib).
 * Copyright (C) 2011 Mitchell Keith Bloch (bazald).
 *
 * zenilib is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * zenilib is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with zenilib.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef ZENI_DATABASE_HXX
#define ZENI_DATABASE_HXX

// HXXed below
#include <Zeni/Resource.h>

#include <Zeni/Database.h>

#include <Zeni/XML.h>

#include <algorithm>
#include <iostream>

namespace Zeni {
  template <class TYPE>
  Database<TYPE>::Lookup::Handle::Handle()
    : ptr(0),
    lent(false)
  {
  }

  template <class TYPE>
  Database<TYPE>::Lookup::Handle::Handle(const String &filename_)
    : ptr(0),
    filename(filename_),
    lent(false)
  {
  }

  template <class TYPE>
  Database<TYPE>::Lookup::Handle::Handle(TYPE * const &ptr_,
                                         const String &filename_,
                                         const bool &lent_,
                                         const bool &keep_)
    : ptr(ptr_),
    filename(filename_),
    lent(lent_),
    keep(keep_)
  {
  }

  template <class TYPE>
  bool Database<TYPE>::Lookup::Handle::operator==(const Handle &rhs) const {
    return filename == rhs.filename;
  }

  template <class TYPE>
  Database<TYPE>::Lookup::Lookup()
    : id(0)
  {
  }

  template <class TYPE>
  Database<TYPE>::Lookup::Lookup(const unsigned long &id_, const Handle &handle_)
    : id(id_)
  {
    handles.push_front(handle_);
  }

  template <class TYPE>
  Database<TYPE>::Database(const String &filename, const String &xml_identifier)
    : m_xml_identifier(xml_identifier),
    m_lost(true)
  {
    m_filenames.push_front(filename);
  }

  template <class TYPE>
  Database<TYPE>::~Database() {
    uninit();
  }
  
  template <class TYPE>
  unsigned long Database<TYPE>::give(const String &name, TYPE * const &type, const bool &keep, const String &filename) {
    if(!type)
      throw Null_Database_Entry_Set();

    Lookup *&lr = m_lookups[name];
    if(!lr)
      lr = new Lookup();

    if(!lr->id)
      lr->id = get_Resource().assign();

    typename Lookup::Handles::iterator it = std::find(lr->handles.begin(), lr->handles.end(), typename Lookup::Handle(filename));

    if(it != lr->handles.end()) {
      if(!it->lent)
        delete it->ptr;
      lr->handles.erase(it);
    }

    lr->handles.push_front(typename Lookup::Handle(type, filename, false, keep));

    m_entries[lr->id] = type;

    return lr->id;
  }
  
  template <class TYPE>
  unsigned long Database<TYPE>::lend(const String &name, TYPE * const &type, const bool &keep) {
    if(!type)
      throw Null_Database_Entry_Set();

    Lookup *&lr = m_lookups[name];
    if(!lr)
      lr = new Lookup();

    if(!lr->id)
      lr->id = get_Resource().assign();

    typename Lookup::Handles::iterator it = std::find(lr->handles.begin(), lr->handles.end(), typename Lookup::Handle(""));

    if(it != lr->handles.end()) {
      if(!it->lent)
        delete it->ptr;
      lr->handles.erase(it);
    }

    lr->handles.push_front(typename Lookup::Handle(type, "", true, keep));

    m_entries[lr->id] = type;

    return lr->id;
  }

  template <class TYPE>
  void Database<TYPE>::clear(const String &name, const String &filename) {
    typename Lookups::iterator it = m_lookups.find(name);

    if(it == m_lookups.end())
      throw Database_Entry_Not_Found("*::" + name);

    Lookup &lr = *it->second;
    typename Lookup::Handles::iterator jt = std::find(lr.handles.begin(), lr.handles.end(), typename Lookup::Handle(filename));

    if(jt == lr.handles.end())
      throw Database_Entry_Not_Found(filename + "::" + name);

    if(!jt->lent)
      delete jt->ptr;

    lr.handles.erase(jt);

    if(lr.handles.empty()) {
      m_entries.erase(lr.id);
      delete it->second;
      m_lookups.erase(it);
    }
    else
      m_entries[lr.id] = lr.handles.begin()->ptr;
  }

  template <class TYPE>
  unsigned long Database<TYPE>::get_id(const String &name) const {
    typename Lookups::const_iterator it = m_lookups.find(name);

    if(it == m_lookups.end() || !it->second->id)
      throw Database_Entry_Not_Found(name);

    return it->second->id;
  }

  template <class TYPE>
  unsigned long Database<TYPE>::find(const String &name) const {
    typename Lookups::const_iterator it = m_lookups.find(name);

    if(it != m_lookups.end() && it->second->id && find(it->second->id))
      return it->second->id;

    return 0;
  }

  template <class TYPE>
  bool Database<TYPE>::find(const unsigned long &id) const {
    return m_entries.find(id) != m_entries.end();
  }

  template <class TYPE>
  TYPE & Database<TYPE>::operator[](const unsigned long &id) const {
    typename Entries::const_iterator it = m_entries.find(id);

    if(it == m_entries.end() || !it->second) {
      char buf[64];
#ifdef _WINDOWS
      sprintf_s
#else
      sprintf
#endif
        (buf, "ID = %u", static_cast<unsigned int>(id));
      throw Database_Entry_Not_Found(buf);
    }

    return *it->second;
  }

  template <class TYPE>
  TYPE & Database<TYPE>::operator[](const String &name) const {
    return (*this)[get_id(name)];
  }

  template <class TYPE>
  void Database<TYPE>::clear() {
    uninit();
  }

  template <class TYPE>
  void Database<TYPE>::load_file(const String &filename) {
    Filenames::iterator it = std::find(m_filenames.begin(), m_filenames.end(), filename);
    if(it != m_filenames.end())
      m_filenames.erase(it);
    m_filenames.push_front(filename);

    XML_Document types_xml(filename.c_str());
    XML_Element_c types = types_xml[m_xml_identifier];
    String name;

    try {
      for(XML_Element_c it = types.first(); it.good(); it = it.next()) {
        name = it.value();

        if(!give_priority(name, false, false, filename)) {
          TYPE * const type = load(it, name, filename);
          if(!type)
            throw Database_Load_Entry_Failed(name);

          give(name, type, false, filename);
        }
      }
    }
    catch(...)
    {
      const String error = "Error loading '" + m_xml_identifier + "' entry '" + name + "'";
      std::cerr << error.c_str() << std::endl;
      message_box(error);
      throw;
    }

    on_load();
  }

  template <class TYPE>
  void Database<TYPE>::unload_file(const String &filename) {
    Filenames::iterator it = std::find(m_filenames.begin(), m_filenames.end(), filename);
    if(it == m_filenames.end())
      throw Database_File_Not_Loaded(filename);

    for(typename Lookups::iterator it = m_lookups.begin();
        it != m_lookups.end();
        ++it)
    {
      for(typename Lookup::Handles::iterator jt = it->second->handles.begin();
          jt != it->second->handles.end();)
      {
        if(jt->filename != filename)
          ++jt;
        else {
          if(!jt->lent)
            delete jt->ptr;
          jt = it->second->handles.erase(jt);
        }
      }

      if(!it->second->handles.empty()) {
        m_entries[it->second->id] = it->second->handles.begin()->ptr;
//         ++it;
      }
      else {
        m_entries.erase(it->second->id);
//         delete it->second;
//         it = m_lookups.erase(it);
      }
    }

    m_filenames.erase(it);

    if(m_filenames.empty())
      on_clear();
  }

  template <class TYPE>
  void Database<TYPE>::reload() {
    lose_resources();
    init();
  }

  template <class TYPE>
  void Database<TYPE>::init() {
    const Filenames old = m_filenames;

    for(Filenames::const_reverse_iterator it = old.rbegin();
       it != old.rend();
       ++it)
    {
      load_file(*it);
    }

    m_lost = false;
  }

  template <class TYPE>
  void Database<TYPE>::uninit() {
    on_clear();

    m_entries.clear();

    for(typename Lookups::iterator it = m_lookups.begin();
        it != m_lookups.end();
        ++it)
    {
      for(typename Lookup::Handles::iterator jt = it->second->handles.begin();
          jt != it->second->handles.end();
          ++jt)
      {
        if(!jt->lent)
          delete jt->ptr;
      }

      delete it->second;
    }

    m_lookups.clear();
  }

  template <class TYPE>
  bool Database<TYPE>::give_priority(const String &name,
                                     const bool &lent,
                                     const bool &keep,
                                     const String &filename)
  {
    const typename Lookups::const_iterator it = m_lookups.find(name);
    if(it == m_lookups.end())
      return false;

    typename Lookup::Handles &lhr = it->second->handles;
    typename Lookup::Handles::iterator jt = std::find(lhr.begin(), lhr.end(), typename Lookup::Handle(filename));
    if(jt == lhr.end())
      return false;

    if(jt->lent != lent || jt->keep != keep)
      return false;

    const typename Lookup::Handle handle = *jt;
    lhr.erase(jt);
    lhr.push_front(handle);

    m_entries[it->second->id] = handle.ptr;

    return true;
  }

  template <class TYPE>
  const bool & Database<TYPE>::lost_resources() {
    return m_lost;
  }

  template <class TYPE>
  void Database<TYPE>::lose_resources() {
    on_lose();

    for(typename Lookups::iterator it = m_lookups.begin();
        it != m_lookups.end();
        ++it)
    {
      for(typename Lookup::Handles::iterator jt = it->second->handles.begin();
          jt != it->second->handles.end();)
      {
        if(jt->keep)
          ++jt;
        else {
          if(!jt->lent)
            delete jt->ptr;
          jt = it->second->handles.erase(jt);
        }
      }

      if(!it->second->handles.empty())
        m_entries[it->second->id] = it->second->handles.begin()->ptr;
      else
        m_entries.erase(it->second->id);
    }

    m_lost = true;
  }

  template <class TYPE>
  void Database<TYPE>::unlose_resources() {
    if(m_lost)
      init();
  }

}

#include <Zeni/Resource.hxx>

#endif
